import numpy as np, sys, math, os

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
def set_gpu_using(memory_rate = 1.0, gpus = '0'):  
    """ 
    This function is to allocate GPU memory a specific fraction 
    """  
    from keras import backend as K
    import tensorflow as tf
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = memory_rate,
            visible_device_list = gpus,
            allow_growth = True,
            )
  
    if num_threads:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options,
                    intra_op_parallelism_threads = num_threads))
    else:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options))
    K.set_session(session)
    return session

class UniqueName:
    def __init__(self):
        self.nc = {}

    def __call__(self, name):
        self.nc.setdefault(name, 0)
        self.nc[name] += 1
        return '{}_{}'.format(name, self.nc[name])
gun = UniqueName()

def gen_idxs(n, m):
    a = np.arange(n)
    while True:
        np.random.shuffle(a)
        for i in range(0, n, m):
            yield a[i: i + m]

def main():
    print('hello world, util.py')

if __name__ == '__main__':
    main()

