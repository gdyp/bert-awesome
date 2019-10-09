# coding: utf-8
import torch

from pytorch_pretrained_bert.modeling import BertModel
from utilss.args import Args
from utilss.utils import tokenizer, cos_sim

args = Args.args
tokenizer = tokenizer()


class SentenceEmbedding(object):
    def __init__(self):
        self.model = BertModel.from_pretrained(args['bert_model'])
        # if torch.cuda.is_available():
        #     self.model.cuda()
        self.model.eval()
        self.input_ids = []
        self.input_mask = []
        self.input_type_id = []

    def build_input(self, sentence):
        tokens = tokenizer.tokenize(sentence)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

        segments_ids = [0] * (len(tokens))
        input_mask = [1] * (len(tokens))

        return tokens_ids, segments_ids, input_mask

    def sentence_embedding(self, sentence):
        token_ids, segment_ids, input_mask = self.build_input(sentence)

        token_ids = torch.LongTensor([token_ids])  # .cuda()
        segment_ids = torch.LongTensor([segment_ids])  # .cuda()
        input_mask = torch.LongTensor([input_mask])  # .cuda()

        with torch.no_grad():
            encoder_layers, _ = self.model(input_ids=token_ids,
                                           token_type_ids=segment_ids,
                                           attention_mask=input_mask,
                                           output_all_encoded_layers=False)


        return encoder_layers[0, -1, :].detach().cpu().numpy().tolist()


if __name__ == '__main__':
    sentence = '今天天气不错'
    sentence_b = '你认识我吗'
    print(tokenizer.tokenize(sentence))
    print(tokenizer.tokenize(sentence_b))
    model = SentenceEmbedding()
    vector_a = model.sentence_embedding(sentence)
    vector_b = model.sentence_embedding(sentence_b)
    print(cos_sim(vector_a, vector_b))
