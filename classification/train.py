# /user/bin/python3.6
# -*-coding: utf-8-*-

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pathlib import Path
import torch
from tqdm import tqdm_notebook as tqdm
import os
from tqdm import tqdm
import sys
import random
import numpy as np
# import apex
from tensorboardX import SummaryWriter
from utils.args import Args
from classification.models import BertForMultiLabelSequenceClassification, CyclicLR
from inputters import MultiLabelTextProcessor, convert_examples_to_features

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA_VISIBLE_DEVICES = 1

DATA_PATH = Path('/data/gump/bert_chinese/data/')
DATA_PATH.mkdir(exist_ok=True)

PATH = Path('/data/gump/bert_chinese/')
PATH.mkdir(exist_ok=True)

CLAS_DATA_PATH = PATH / 'class'
CLAS_DATA_PATH.mkdir(exist_ok=True)

OUTPUT_DIR = Path('/data/gump/bert_chinese/model/')
model_state_dict = None

BERT_PRETRAINED_PATH = Path('/data/gump/bert_chinese/chinese_L-12_H-768_A-12/')

PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH / 'cache/'
PYTORCH_PRETRAINED_BERT_CACHE.mkdir(exist_ok=True)

args = Args(full_data_dir=DATA_PATH, data_dir=PATH, bert_model=BERT_PRETRAINED_PATH,
            output_dir=OUTPUT_DIR).args

processors = {
    "intent_multilabel": MultiLabelTextProcessor
}

if args["local_rank"] == -1 or args["no_cuda"]:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    # n_gpu = torch.cuda.device_count()
    n_gpu = 1
else:
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if n_gpu > 0:
    torch.cuda.manual_seed_all(args['seed'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name](args['data_dir'])
# label_list = processor.get_labels()
label_list = ['alarm', 'bag', 'chat', 'command', 'face', 'greet', 'intelligent_home', 'machine', 'food',
              'music', 'news', 'query', 'radio', 'sleep', 'story', 'time', 'volume', 'weather', 'study']
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args['bert_model'])

train_examples = None
num_train_steps = None
if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(
        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])

#     pdb.set_trace()
if model_state_dict:
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        '/data/gump/bert_chinese/chinese_L-12_H-768_A-12', num_labels=num_labels, state_dict=model_state_dict)
else:
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        '/data/gump/bert_chinese/chinese_L-12_H-768_A-12', num_labels=num_labels)

if args['fp16']:
    model.half()

model.to(device)

if args['local_rank'] != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

    # Prepare optimizer

model.unfreeze_bert_encoder()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
t_total = num_train_steps
if args['local_rank'] != -1:
    t_total = t_total // torch.distributed.get_world_size()
if args['fp16']:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args['learning_rate'],
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args['loss_scale'] == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total)

for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v): state[k] = v.cuda(device)

scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


# Eval Fn
eval_examples = processor.get_dev_examples(args['full_data_dir'], size=args['val_size'])

# writer = SummaryWriter()


# eval_examples = processor.get_test_examples('/home/data/peter/intent_classification_code/intent_classification/KFold_data/',
#                                            'test.csv', size=args['val_size'])


def eval(epoch):
    args['output_dir'].mkdir(exist_ok=True)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    all_logits = None
    all_labels = None

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.view(-1).to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        #         tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        #         if all_logits is None:
        #             all_logits = logits.detach().cpu().numpy()
        #         else:
        #             all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        #         if all_labels is None:
        #             all_labels = label_ids.detach().cpu().numpy()
        #         else:
        #             all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    #     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()

    #     for i in range(num_labels):
    #         fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])

    #     # Compute micro-average ROC curve and ROC area
    #     fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    #               'loss': tr_loss/nb_tr_steps}
    #               'roc_auc': roc_auc  }

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))
    if epoch>5:
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model" + str(epoch) + ".bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    return result


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


train_features = convert_examples_to_features(
    train_examples, label_list, args['max_seq_length'], tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args['train_batch_size'])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args['local_rank'] == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])


def fit(num_epocs=args['num_train_epochs']):
    global_step = 0
    model.train()
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='train')):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_ + 1))

        if i_ > 5:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                             "finetuned_pytorch_model" + str(i_) + ".bin")
            torch.save(model_to_save.state_dict(), output_model_file)
        # result = eval(epoch=i_)
        writer.add_scalar('scalar/loss', tr_loss / nb_tr_steps, i_)
        # writer.add_scalar('scalar/loss', result['eval_loss'], i_)
        # writer.add_scalar('scalar/acc', result['eval_accuracy'], i_)


# model.unfreeze_bert_encoder()

if __name__ == '__main__':
    writer = SummaryWriter()
    fit()
    writer.close()
    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
    # torch.save(model_to_save.state_dict(), output_model_file)
