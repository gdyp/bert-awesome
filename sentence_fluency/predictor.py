#! -*- coding: utf-8 -*-
import os

import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

from sentence_fluency.processing import MAX_LENGTH
from utilss.utils import features_translation, tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SOURCE_PATH = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/' \
              'sentence_fluency/data/'
CONFIG_FILE = SOURCE_PATH+'model/config_7.json'
MODEL_FILE = SOURCE_PATH+'model/model_7.bin'

# load model
config = BertConfig(CONFIG_FILE)
model = BertForSequenceClassification(config, num_labels=2)
model.load_state_dict(torch.load(MODEL_FILE))
if torch.cuda.is_available():
    model.cuda()
model.eval()


tokenizer = tokenizer()


def predict(sentence):
    token_a = tokenizer.tokenize(sentence)

    token_text = ['[CLS]'] + token_a + ['[SEP]']
    tokens_ids = tokenizer.convert_tokens_to_ids(token_text)

    segments_ids = [0] * (len(token_a) + 2)
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

    score = logits.cpu().numpy().tolist()
    return 0 if score[0] > score[1] else 1
#
# if __name__ == '__main__':
#     sentence = '你是谁'
#     result = predict(sentence)
#     print(result)

