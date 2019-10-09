#! -*- coding: utf-8 -*-
import csv
import pandas as pd
from tqdm import tqdm

import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertForNextSentencePrediction

# from next_sentence.processing import create_features, read_file, MAX_LENGTH
from utils.utils import features_translation, tokenizer

SOURCE_PATH = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/' \
              'next_sentence/data/'
CONFIG_FILE = SOURCE_PATH+'model/model_2.2.2/config_4.json'
MODEL_FILE = SOURCE_PATH+'model/model_2.2.2/model_4.bin'
TEST_FILE = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/' \
            'next_sentence/data/chat_bot_2.2.0.csv'

MAX_LENGTH = 30
# load model
config = BertConfig(CONFIG_FILE)
model = BertForNextSentencePrediction(config)
model.load_state_dict(torch.load(MODEL_FILE))
if torch.cuda.is_available():
    model.cuda()
model.eval()

# load data
tokenizer = tokenizer()


def predict(text_a, text_b):
    token_a = tokenizer.tokenize(text_a)
    token_b = tokenizer.tokenize(text_b)

    token_text = ['[CLS]'] + token_a + ['[SEP]'] + token_b + ['[SEP]']
    tokens_ids = tokenizer.convert_tokens_to_ids(token_text)

    segments_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) + 1)
    input_mask = [1] * len(token_text)

    assert len(segments_ids) == len(token_text)
    assert len(input_mask) == len(token_text)

    if len(token_text) > MAX_LENGTH:
        return ''

    padding = [0] * (MAX_LENGTH - len(token_text))
    tokens_ids += padding
    segments_ids += padding
    input_mask += padding

    assert len(tokens_ids) == MAX_LENGTH

    with torch.no_grad():
        logits = model(torch.cuda.LongTensor([tokens_ids]),
                       torch.cuda.LongTensor([segments_ids]),
                       torch.cuda.LongTensor([input_mask])).view(-1,).sigmoid()

    score = logits.cpu().numpy()pycharmpppp\
        .tolist()
    return score[0]

