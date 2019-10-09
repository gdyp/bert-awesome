#! -*- coding: utf-8 -*-
"""
sentence fluency test
Precision: 0.93
recall: 0.94
F1-score: 0.88
"""
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

from sentence_fluency.predictor import predict

source = '/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/sentence_fluency/data/'
test_data = pd.read_csv(source+'test.csv')
y_true = [int(label) for label in test_data['label']]
y_pred = []

sentences = test_data['sentence']
for index, sentence in tqdm(enumerate(sentences), total=len(sentences)):
    try:
        y_pred.append(predict(sentence))
    except:
        del y_true[index]
        print(sentence)
assert len(y_pred) == len(y_true)
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print('accuracy:{}, recall:{}, f1_score:{}'.format(accuracy, recall, f1))

# data = open('/home/gump/Software/pycharm-2018.1.6/projects/bert-for-classificaion/sentence_fluency/'
#             'data/test.csv')
# reader = csv.reader(data)
# next(reader)
# num = 0
# with open(source+'error.csv', 'w', encoding='utf_8_sig') as f:
#     writer = csv.writer(f)
#     writer.writerow(['sentence', 'label'])
#     for line in tqdm(reader, total=12716):
#         sentence = line[0].strip()
#         result = predict(sentence)
#         if result != int(line[1]):
#             print(line)
#             writer.writerow(line)
