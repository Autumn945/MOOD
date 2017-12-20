import numpy as np, sys, math, os
from keras.layers import Layer
from keras import backend as K

class PosAddW(Layer):
    def __init__(self, l2, *args, **kwargs):
        self.l2 = l2
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = tuple(input_shape[1:])
        self.w = self.add_weight('posw', shape, regularizer = self.l2, initializer = 'uniform')
        super().build(input_shape)

    def call(self, x, mask = None):
        return x + self.w

    def compute_mask(self, inputs, mask = None):
        return mask

def main():
    print('hello world, MyLayers.py')

if __name__ == '__main__':
    main()

