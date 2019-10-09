#! -*- coding: utf-8 -*-
class InputExample(object):

    def __init__(self, guid=None, text_a=None, text_b=None, labels=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()