#! -*- coding: utf-8 -*-
import csv
import torch
import pickle
from tqdm import tqdm

from utils.utils import tokenizer
from config import BERT_PRETRAINED_PATH
from inputters import InputExample, InputFeatures


MAX_LENGTH = 30
tokenizer = tokenizer()

number_map = {
    '0': 1,
    '1': 0
}


def read_file(path):
    """csv file"""
    file = open(path)
    reader = csv.reader(file)
    next(reader)  # filter header
    for line in reader:
        line[2] = number_map[line[2]]
        yield line


def create_examples(data):
    examples = []

    for i, line in enumerate(data):
        examples.append(InputExample(guid=i, text_a=line[0],
                                     text_b=line[1], labels=line[2]))
    return examples


def convert_example_to_features(examples):
    features = []

    for line in tqdm(examples, total=len(examples), desc='create examples'):
        token_a = tokenizer.tokenize(line.text_a)
        token_b = tokenizer.tokenize(line.text_b)

        token_text = ['[CLS]']+token_a+['[SEP]']+token_b+['[SEP]']
        tokens_ids = tokenizer.convert_tokens_to_ids(token_text)

        segments_ids = [0]*(len(token_a)+2)+[1]*(len(token_b)+1)
        input_mask = [1]*len(token_text)

        assert len(segments_ids) == len(token_text)
        assert len(input_mask) == len(token_text)

        if len(token_text) > MAX_LENGTH:
            continue

        padding = [0]*(MAX_LENGTH-len(token_text))
        tokens_ids += padding
        segments_ids += padding
        input_mask += padding

        assert len(tokens_ids) == MAX_LENGTH

        features.append(InputFeatures(input_ids=tokens_ids,
                                      input_mask=input_mask,
                                      segment_ids=segments_ids,
                                      label_ids=int(line.labels)))

    return features


def create_features(path):
    file = read_file(path)
    examples = create_examples(file)
    return convert_example_to_features(examples)


# if __name__ == '__main__':
#     file_path = '/home/gump/Software/pycharm-2018.1.6/' \
#                 'projects/bert-for-classificaion/next_sentence/data/train.csv'
#     file = read_file(file_path)
#     examples = create_examples(file)
#     feature = convert_example_to_features(examples)
#
#     with open('val_features.pkl', 'wb') as f:
#         pickle.dump(examples, f)
#         # examples = pickle.load(f)
#     # for i in range(10):
#     #     print('success')


