#! -*- coding: utf-8 -*-
import sys
import os
import csv
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/')
from inputters import InputExample, InputFeatures, DataProcessor
from utilss.utils import tokenizer


MAX_LENGTH = 50
tokenizer = tokenizer()


class Processing(DataProcessor):

    def get_labels(self):
        pass

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_train_examples(self):
        filename = 'train.csv'
        data = self._read_file(os.path.join(self.data_dir, filename))
        return self._create_examples(data)

    def get_dev_examples(self):
        filename = 'val.csv'
        data_df = self._read_file(os.path.join(self.data_dir, filename))
        return self._create_examples(data_df)

    def get_test_examples(self):
        data_df = self._read_file(os.path.join(self.data_dir, 'test.csv'))
        return self._create_examples(data_df)

    def _read_file(self, path):
        """csv file"""
        data = open(path)
        reader = csv.reader(data)
        next(reader)
        for line in reader:
            yield line
        data.close()

    def _create_examples(self, data):
        examples = []

        for i, line in enumerate(data):
            examples.append(InputExample(guid=i, text_a=line[0], labels=line[1]))
        return examples


def convert_example_to_features(examples):
    features = []

    for line in tqdm(examples, total=len(examples), desc='create examples'):
        token_a = tokenizer.tokenize(line.text_a)

        token_text = ['[CLS]']+token_a+['[SEP]']
        tokens_ids = tokenizer.convert_tokens_to_ids(token_text)

        segments_ids = [0]*(len(token_a)+2)
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
                                      label_id=int(line.labels)))

    return features
