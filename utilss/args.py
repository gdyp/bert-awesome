#! -*- coding: utf-8 -*-
class Args:
    args = {
        "data_dir": None,
        "bert_model": '/data/gump/bert_chinese/chinese_L-12_H-768_A-12',
        "task_name": "intent_multilabel",
        "output_dir": None,

        "cache_dir": None,
        "max_seq_length": 128,
        "do_train": True,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": None,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "num_train_epochs": None,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 0,
        "server_ip": '',
        "server_port": '',
        "layers": -1
    }


if __name__ == '__main__':
    args = Args.args
    print(args)