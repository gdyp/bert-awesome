#! -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from config import BERT_PRETRAINED_PATH
from processing import create_features
from utils.utils import features_translation

SOURCE = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/next_sentence/'
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default=SOURCE+'data/train.csv')
parser.add_argument('--val_file', default=SOURCE+'data/val.csv')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--lr', default=3e-5)
parser.add_argument('--epochs', default=20)
parser.add_argument('--fp16', default=False)
parser.add_argument('--warmup_proportion', default=0.1)
parser.add_argument('--seed', default=42)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--loss_scale', type=float, default=1)
parser.add_argument('--output_dir', default=SOURCE+'data/model/model_2.2.2/')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


args.batch_size = args.batch_size // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# load data
logging.info('create train features')
num_train_optimization_steps = None
train_features = create_features(getattr(args, 'train_file'))
train_features_data = features_translation(train_features)

num_train_optimization_steps = int(len(train_features)/args.batch_size/args.gradient_accumulation_steps)*args.epochs

logging.info('create validation features')
val_features = create_features(getattr(args, 'val_file'))
val_features_data = features_translation(val_features)

logging.info('create batch data')
train_data = DataLoader(train_features_data, batch_size=getattr(args, 'batch_size'),
                        shuffle=True, drop_last=True)
val_data = DataLoader(val_features_data, batch_size=getattr(args, 'batch_size'),
                        shuffle=True, drop_last=True)

# load model
logging.info('create model')
model = BertForNextSentencePrediction.from_pretrained(BERT_PRETRAINED_PATH, cache_dir='data/cache')
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
                         lr=args.lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

# train
global_step = 0
last_val_loss = 100
epochs = getattr(args, 'epochs')
for i in range(1, epochs+1):
    training_loss = 0

    model.train()
    for step, batch in enumerate(tqdm(train_data, desc='train', total=len(train_data))):
        if torch.cuda.is_available():
            batch = tuple(item.cuda() for item in batch)
        input_ids, segment_ids, input_mask, label_ids = batch

        loss = model(input_ids, segment_ids, input_mask, label_ids)
        training_loss += loss.item()

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step+1)%args.gradient_accumulation_steps == 0:
            if args.fp16:
                lr_this_step = args.lr*warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    validation_loss = 0
    model.eval()
    for batch in val_data:
        if torch.cuda.is_available():
            batch = (item.cuda() for item in batch)
        input_ids, segment_ids, input_mask, label_ids = batch

        with torch.no_grad():
            loss = model(input_ids, segment_ids, input_mask, label_ids)
        validation_loss += loss.item()

    training_loss = training_loss/len(train_data)
    validation_loss = validation_loss/len(val_data)
    logging.info('{}/{}, train loss: {}, validation loss: {}'.format(i, epochs, training_loss, validation_loss))

    if validation_loss < last_val_loss:
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, 'model_'+str(i)+'.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config_'+str(i)+'.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    last_val_loss = validation_loss


