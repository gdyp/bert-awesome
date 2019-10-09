#! -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import TensorDataset


from pytorch_pretrained_bert import BertTokenizer
from utilss.args import Args

args = Args.args


def tokenizer():
    return BertTokenizer.from_pretrained(args.get('bert_model', '/data/gump/bert_chinese/chinese_L-12_H-768_A-12'))


def features_translation(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    return data


def get_accuracy(logits, labels, correct_count):
    for i in range(logits.size()[0]):
        if (logits.data[i][0] > logits.data[i][1] and labels.data[i] == 0) or \
                (logits.data[i][0] < logits.data[i][1] and labels.data[i] == 1):
            correct_count += 1

    return correct_count


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)

    num = float(vector_a*vector_b.T)
    denom = np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    cos = num / denom

    return cos

if __name__ == '__main__':
    token = tokenizer()
    print(token.tokenize('今天天气不错'))
