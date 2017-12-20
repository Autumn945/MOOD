import numpy as np
import sys
import math
import os
import time
import pandas as pd
import json
import logger

class Data:
    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y

class BasicData:
    def __init__(self, dataset, maxlen = 100):
        self.ds = dataset
        self.maxlen = maxlen
        fn = 'data/{}.json'.format(dataset)
        self.data = json.load(open(fn, 'r'))
        self.__dict__.update(self.data)
        self.parse_data()

    def preprocessing(self):
        pass

    def parse_data(self):
        self.preprocessing()
        for name in 'train,vali,test'.split(','):
            data = self.__dict__[name]
            data = Data(data, self.get_x(data), self.get_y(data))
            self.__dict__[name] = data
    
    def get_x(self, data):
        return [1] * len(data)

    def get_y(self, data):
        label = np.array(data['label'], dtype = 'float32')
        label = np.log(label + 1)
        return label

class kerasData(BasicData):
    @logger.run_time
    def pad_text(self, texts):
        from keras.preprocessing.sequence import pad_sequences
        text = pad_sequences(texts, maxlen = self.maxlen, padding = 'post', truncating = 'post')
        return text

    @logger.run_time
    def get_x(self, data):
        user = np.array(data['user']).reshape((-1, 1))
        loc = np.array(data['location']).reshape((-1, 1))
        texts = data['text']
        text = self.pad_text(texts)
        return [user, loc, text]

def main():
    print('hello world, dataset')

if __name__ == '__main__':
    main()

