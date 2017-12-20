import numpy as np
import sys
import math
import os
import random
import json
import time
import argparse

from dataset import kerasData
import logger
from logger import log
from pprint import pformat
import keras_models

np.random.seed(171103)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

@logger.run_time
def run(dataset, maxlen = 100, **kwargs):
    data = kerasData(dataset = dataset, maxlen = maxlen)
    log('dataset: {}, maxlen: {}, gpu: {}'.format(
        dataset,
        maxlen,
        kwargs['gpu']
        ))
    log('#train: {}, #vali: {}, #test: {}'.format(
        len(data.train.y),
        len(data.vali.y),
        len(data.test.y)
        ))
    log(pformat(kwargs))
    model = keras_models.MOOD(
            data = data,
            maxlen = maxlen,
            **kwargs)
    args = get_args()
    ve, te = model.fit(args)
    print(ve, te)

def get_args():
    args = {}
    args['dense_l2'] = 1e-2
    args['dense_b_l2'] = 1e-2

    args['att_user_l2'] = 0
    args['att_loc_l2'] = 1e-2
    args['pair_user_l2'] = 1e-5
    args['pair_loc_l2'] = 1e-4
    args['bias_user_l2'] = 0
    args['bias_loc_l2'] = 1e-2

    args['att_text_l2'] = 1e-4
    args['att_text_pos_l2'] = 1e-3
    args['mem_text_l2'] = 0
    args['mem_text_pos_l2'] = 1e-3

    args['nb_mem_layers'] = 2
    return args

@logger.run_time
def main():
    print('hello world, main')
    parser = argparse.ArgumentParser(description = "MOOD's args")

    def add_argument(*v, **kw):
        parser.add_argument(*v, **kw)

    add_argument('-ds', '--dataset', type = str, default = 'shanghai')
    add_argument('-gpu', type = str, default = '3')
    add_argument('-rt', '--run_times', type = int, default = 1)
    add_argument('-es', '--early_stop', type = int, default = 5)
    add_argument('-k', '--dim_k', type = int, default = 128)
    add_argument('-bsteps', '--batch_steps', type = int, default = -1)
    add_argument('-bsize', '--batch_size', type = int, default = 128)
    add_argument('-me', '--max_epochs', type = int, default = 200)

    args = parser.parse_args()
    kw = args.__dict__
    run(**kw)

if __name__ == '__main__':
    main()

