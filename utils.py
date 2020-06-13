from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow import set_random_seed
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
set_random_seed(3)
np.random.seed(3)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import random

def batch_generator_files(filenames, batch_size, shuffle=True):
    data_dir = '/DATA/public/bladder_40_40_32'
    if shuffle:
        random.shuffle(filenames)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(filenames):
            batch_count = 0
            if shuffle:
                random.shuffle(filenames)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        output = [np.zeros((batch_size,40,40,32,1)), np.zeros((batch_size,40,40, 32))]
        for i in range(batch_size):
            output[0][i,:,:,:,0] = np.load(data_dir + '/' + filenames[start+i] + '-image.npy')
            output[1][i,] = np.load(data_dir + '/' + filenames[start+i] + '-mask.npy')
        yield output

def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) +'_'
    results_name = results_name[:-1]
    return results_name