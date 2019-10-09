#! -*- coding: utf-8 -*-

import torch

from utils.utils import tokenizer
from inputters import InputFeatures
from classification.models import BertForMultiLabelSequenceClassification

LABELS = ['alarm', 'bag', 'chat', 'command', 'face', 'greet', 'intelligent_home', 'machine', 'food',
          'music', 'news', 'query', 'radio', 'sleep', 'story', 'time', 'volume', 'weather', 'study']
BERT_MODEL = '/data/gump/bert_chinese/chinese_L-12_H-768_A-12'
NUM_LABELS = 19
STATE_DICT = '/data/gump/bert_chinese/chinese_L-12_H-768_A-12/cache/finetuned_pytorch_model14.bin'

state_dict = torch.load(STATE_DICT, map_location=lambda storage, loc: storage)


class Predict(object):
    def __init__(self):
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS,
                                                                             state_dict=state_dict)
        # if torch.cuda.is_available():
        #     self.model.cuda()
        self.max_length = 18
        self.label_map = {i: label for i, label in enumerate(LABELS)}
        self.tokenizer = tokenizer()

    def get_features(self, sentence):
        features = []
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=None))
        return features

    def predict(self, sentence):
        features = self.get_features(sentence)
        input_ids = torch.tensor([item.input_ids for item in features], dtype=torch.long)  # .cuda()
        input_mask = torch.tensor([item.input_mask for item in features], dtype=torch.long)  # .cuda()
        segment_ids = torch.tensor([item.segment_ids for item in features], dtype=torch.long)  # .cuda()

        with torch.no_grad():
            logits = self.model(input_ids, input_mask, segment_ids).view(-1,).sigmoid()

        score = {self.label_map[i]: item for i, item in enumerate(logits.tolist())}

        score = sorted(score.items(), key=lambda x: x[1], reverse=True)
        return score[:3]


if __name__ == '__main__':
    sentences = '你去学习吧'
    result = Predict().predict(sentences)
    print(result)
