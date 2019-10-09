#! -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm
import random
import numpy as np
import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, DistributedSampler
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from utils.args import Args
from utils.utils import get_accuracy
from sentence_fluency.processing import Processing, convert_example_to_features

SOURCE = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/sentence_fluency/data/'
args_dict = Args.args
args_dict['data_dir'] = SOURCE
args_dict['train_batch_size'] = 64
args_dict['output_dir'] = SOURCE + 'model/'
args_dict['learning_rate'] = 5e-5
args_dict['num_train_epochs'] = 7

parser = argparse.ArgumentParser()
for key, value in args_dict.items():
    parser.add_argument('-'+key, default=value)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
processor = Processing(data_dir=args.data_dir)

# load data
logging.info('create train features')
num_train_optimization_steps = None
train_examples = processor.get_train_examples()
train_features = convert_example_to_features(train_examples)
# train_features_data = features_translation(train_features)

all_input_ids = torch.LongTensor([f.input_ids for f in train_features])
all_input_mask = torch.LongTensor([f.input_mask for f in train_features])
all_segment_ids = torch.LongTensor([f.segment_ids for f in train_features])
all_label_ids = torch.LongTensor([f.label_id for f in train_features])
# num_train_optimization_steps = int(len(train_features)/args.train_batch_size/args.gradient_accumulation_steps)*args.epochs
if args.do_train:
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

logging.info('create batch data')
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

val_examples = processor.get_dev_examples()
val_features = convert_example_to_features(val_examples)

val_input_ids = torch.LongTensor([f.input_ids for f in train_features])
val_input_mask = torch.LongTensor([f.input_mask for f in train_features])
val_segment_ids = torch.LongTensor([f.segment_ids for f in train_features])
val_label_ids = torch.LongTensor([f.label_id for f in train_features])

val_data = TensorDataset(val_input_ids, val_input_mask, val_segment_ids, val_label_ids)

val_dataloader = DataLoader(val_data, shuffle=True, batch_size=args.eval_batch_size)
# load model
logging.info('create model')
model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir='data/cache', num_labels=2)
if args.fp16:
    model.half()
if torch.cuda.is_available():
    model.cuda()

# optimizer
parameters = list(model.named_parameters())
# parameters = [n for n in parameters if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError('please install apex')

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=getattr(args, 'lr'),
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
loss_fct = CrossEntropyLoss()
# train
global_step = 0
last_val_loss = 100
epochs = getattr(args, 'num_train_epochs')
for i in range(1, epochs + 1):
    training_loss = 0

    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc='train', total=len(train_dataloader))):
        if torch.cuda.is_available():
            batch = tuple(item.cuda() for item in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        logits = model(input_ids, segment_ids, input_mask)

        loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        training_loss += loss.item()

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                lr_this_step = args.lr * warmup_linear(global_step / num_train_optimization_steps,
                                                       args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    validation_loss = 0
    correct_count = 0
    model.eval()
    for batch in val_dataloader:
        if torch.cuda.is_available():
            batch = (item.cuda() for item in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            val_logits = model(input_ids, segment_ids, input_mask)

        val_loss = loss_fct(val_logits.view(-1, 2), label_ids.view(-1))
        correct_count = get_accuracy(val_logits.view(-1, 2), label_ids.view(-1), correct_count)
        validation_loss += val_loss.item()

    training_loss = training_loss / len(train_dataloader)
    validation_loss = validation_loss / len(val_data)
    accuracy = correct_count / len(val_data)
    logging.info('{}/{}, train loss: {}, validation loss: {}, val_accuracy: {}'.format(
        i, epochs, training_loss, validation_loss, accuracy))

    if validation_loss < last_val_loss:
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, 'model_' + str(i) + '.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config_' + str(i) + '.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    last_val_loss = validation_loss
