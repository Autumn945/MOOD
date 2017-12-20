import numpy as np, sys, math, os
import copy
import time
import keras
from keras import backend as K
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import TimeDistributed

from keras.layers import LSTM, GRU
from keras.layers import Bidirectional

from keras.layers import concatenate
from keras.layers import add
from keras.layers import multiply

from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences

from collections import OrderedDict

import logger
from logger import log, _timer
import util
import AttLayers
import MyLayers
from evaluate import Eva
import dataset

class BasicModel:
    _fields = ['data', 'run_times', 'maxlen', 'early_stop', 'max_epochs', 'dim_k', 'batch_size', 'batch_steps', 'gpu']
    def __init__(self, *args, **kwargs):
        self.name = type(self).__name__
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))

        if self.batch_steps < 0:
            self.batch_steps = math.ceil(len(self.data.train.y) / self.batch_size)
        self.early_stop *= math.ceil(len(self.data.train.y) / (self.batch_size * self.batch_steps))
        log('early_stop: {}'.format(self.early_stop))
        n = len(self.data.train.y)
        self.epp = math.ceil(n / (self.batch_size * self.batch_steps))
        log('#train_data: {}\nbatch_size: {}\nsteps: {}\nall data need {} epochs'.format(n, self.batch_size, self.batch_steps, self.epp))
        self.att_values = []
        self.model = None

    def fit(self, model_args):
        seeds = [171204,402711,711270,201714,141702]
        ve, te = [], []
        for _ in range(self.run_times):
            log('run_times#{}/{}, seed: {}'.format(_ + 1, self.run_times, seeds[_]))
            np.random.seed(seeds[_])
            self.init_model_args(model_args)
            self.model = self.make_model()
            _ve, _te = self._fit()
            ve.append(_ve)
            te.append(_te)
            msg = 'vali: {}, test: {}'.format(_ve, _te)
            log(msg)
        return Eva.mean(ve), Eva.mean(te)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y = None):
        if y is None:
            if type(x) == str:
                x = eval('self.data.' + x)
            x, y = x.x, x.y
        return Eva(self.predict(x), y)

    def init_model_args(self, model_args):
        K.clear_session()
        util.set_gpu_using(gpus = self.gpu)
        self.un = util.UniqueName()
        self.model_args = model_args

    def get_l2(self, name):
        all_l2 = self.model_args.get('all_l2')
        if all_l2 is not None:
            return regularizers.l2(all_l2)
        return regularizers.l2(self.model_args['{}_l2'.format(name)])

    def train_on_batch(self, data, batch_size, steps):
        n = len(data.y)
        idxs = util.gen_idxs(n, batch_size)
        x, y = data.x, data.y
        while True:
            if self.epp > 1 or 1:
                for i in range(steps):
                    idx = next(idxs)
                    self.model.train_on_batch([_x[idx] for _x in x], y[idx])
            else:
                self.model.fit(x, y, batch_size = batch_size, epochs = 1, verbose = 0)
            yield 1

    def _fit(self):
        best_weights, best_e, brk = None, None, 0
        training = self.train_on_batch(self.data.train, self.batch_size, self.batch_steps)
        for i in range(self.max_epochs):
            _timer.start()
            next(training)
            train_time = _timer.stop()
            _timer.start()
            e = self.evaluate('vali')
            vali_time = _timer.stop()
            if e.is_better_than(best_e):
                best_e = e
                best_w = self.model.get_weights()
                brk = 0
            else:
                brk += 1
            msg = '#{}/{}, vali: {}, time: {:.2f}s {:.2f}s'.format(i + 1, self.max_epochs, e, train_time, vali_time)
            log(msg)
            if self.early_stop > 0 and brk >= self.early_stop:
                break
        _timer.start()
        self.model.set_weights(best_w)
        test_e = self.evaluate('test')
        _timer.stop()
        return best_e, test_e

    def single_embedding(self, x, n, name, dim_k = None):
        if dim_k is None:
            dim_k = self.dim_k
        x = Embedding(
                n,
                dim_k,
                name = self.un(name),
                embeddings_regularizer = self.get_l2(name),
                )(x)
        x = Flatten()(x)
        return x

    def get_text_emb_layer(self, name = 'text'):
        emb = Embedding(
                self.data.nb_words + 1,
                self.dim_k,
                name = self.un(name),
                embeddings_regularizer = self.get_l2(name),
                mask_zero = True,
                )
        pos = MyLayers.PosAddW(l2 = self.get_l2('{}_pos'.format(name)))
        return emb, pos

    def text_embs(self, text = None, emb_layer = None, name = 'text'):
        if text is None:
            text = self.inp_text
        if emb_layer is None:
            emb_layer = self.get_text_emb_layer(name = name)
        texts = emb_layer[0](text)
        texts = emb_layer[1](texts)
        return texts

    def user_emb(self, name = 'user'):
        return self.single_embedding(self.inp_user, self.data.nb_users, name)

    def loc_emb(self, name = 'loc'):
        return self.single_embedding(self.inp_loc, self.data.nb_locs, name)

    def post_emb(self):
        pass

    def make_inputs(self):
        self.inp_user = Input((1,), name = 'inp_user')
        self.inp_loc = Input((1,), name = 'inp_loc')
        self.inp_text = Input((self.maxlen,), name = 'inp_text')
        self.inputs = [self.inp_user, self.inp_loc, self.inp_text]

    def top_layers(self, x):
        user = self.user_emb('bias_user')
        loc = self.loc_emb('bias_loc')
        x = concatenate([x, user, loc])
        x = Dense(
                512,
                activation = 'relu',
                kernel_regularizer = self.get_l2('dense'),
                bias_regularizer = self.get_l2('dense_b')
                )(x)
        return x

    def make_model(self):
        self.make_inputs()
        x = self.post_emb()
        x = self.top_layers(x)
        y = Dense(
                1,
                activation = 'relu',
                kernel_regularizer = self.get_l2('dense'),
                bias_regularizer = self.get_l2('dense_b')
                )(x)
        model = Model(inputs = self.inputs, outputs = y)
        model.compile(
                optimizer = 'adagrad',
                loss = 'mse')
        return model

class MemNN(BasicModel):
    def mem(self, mem, att, u, l):
        self.att_values = []
        for i in range(self.model_args['nb_mem_layers']):
            att_layer = AttLayers.Att()
            o = att_layer([mem, att, u, l])
            self.att_values.append(att_layer.att_value)
            if i == self.model_args['nb_mem_layers'] - 1:
                return add([o, u, l])
            u = add([u, o])
            l = add([l, o])

    def post_emb(self):
        user = self.user_emb('att_user')
        loc = self.loc_emb('att_loc')
        mem = self.text_embs(name = 'mem_text')
        att = self.text_embs(name = 'att_text')
        return self.mem(mem, att, user, loc)

class MOOD(MemNN):
    def post_emb(self):
        text = super().post_emb()
        user = self.user_emb('pair_user')
        loc = self.loc_emb('pair_loc')
        ut = multiply([user, text])
        lt = multiply([loc, text])
        ul = multiply([user, loc])
        return add([ut, lt, ul])


def main():
    print('hello world')

if __name__ == '__main__':
    main()
