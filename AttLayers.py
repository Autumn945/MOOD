import numpy as np, sys, math, os
from keras.layers import Layer
from keras import backend as K

class Avg(Layer):
    def __init__(self, return_sequences = False, **kw):
        self.return_sequences = return_sequences
        super().__init__(**kw)

    def call(self, x, mask = None):
        if type(x) == list:
            x = x[0]
            mask = mask[0]
        # x: (None, maxlen, dim_k), mask: (None, maxlen)
        if mask is not None:
            mask = K.cast(mask, 'float32')
            x = x * K.expand_dims(mask, -1)
            x = K.sum(x, -2) / K.expand_dims(K.sum(mask, -1), -1)
            return x
        else:
            return K.mean(x, -2)

    def att_output(self, c, att, mask = None):
        att = K.softmax(att)
        if mask is not None:
            att = att * K.cast(mask, 'float32')
            att = att / K.expand_dims(K.sum(att, -1), -1)

        self.att_value = att
        att_text = c * K.expand_dims(att, -1)
        if self.return_sequences:
            return att_text
        return K.sum(att_text, -2)

    def compute_mask(self, inputs, mask = None):
        if type(mask) == list:
            mask = mask[0]
        if self.return_sequences:
            return mask
        return None

    def compute_output_shape(self, inputs):
        if type(inputs) == list:
            inputs = inputs[0]
        if self.return_sequences:
            return inputs
        return (inputs[0], inputs[-1])

class Att(Avg):
    def call(self, x, mask = None):
        m, a, u, l = x
        f = u + l
        att = K.batch_dot(a, f, (2, 1))
        return self.att_output(m, att, mask[0])

def main():
    print('hello world, AttLayers.py')

if __name__ == '__main__':
    main()

