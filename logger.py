import numpy as np, sys, math, os
import time, threading
import functools

indent = 2
func_deep = 0

#wraps
def run_time(func):
    functools.wraps(func)
    def __wrap(*args, **kw):
        global func_deep
        print('{}> call {}()...'.format(' ' * func_deep * indent, func.__name__))
        bt = time.time()
        func_deep += 1
        ret = func(*args, **kw)
        func_deep -= 1
        ct = time.time() - bt
        print('{}< call {}() over, time: {:.2f}'.format(' ' * func_deep * indent, func.__name__, ct))
        return ret
    return __wrap

def run_for(n = None, step = 0):
    def __d(func):
        def __w(*args):
            d = args[-1]
            if step == 0:
                return [func(*args[:-1], _d) for _d in d]
            bt = time.time()
            _n = len(d) if n is None else n
            def _f(i, _d):
                if i % step == 0:
                    nt = time.time() - bt
                    rt = (_n - i) * nt / i if i else 0
                    print('\r{} #{}/{}, cost {:.2f}s, rest: {:.2f}s({:.1f}min),  '.format(func.__name__, i, _n, nt, rt, rt / 60), end = '', flush = True)
                return func(*args[:-1], _d)
            ret = [_f(i, _d) for i, _d in enumerate(d)]
            print('over ~')
            return ret
        return __w
    return __d

def red_str(s):
    return '\033[1;31;40m{}\033[0m'.format(s)

def log(s, **kw):
    s = str(s)
    print(s, **kw)

class Timer:
    def __init__(self):
        self.end = True

    def loop(self):
        while not self.end:
            time.sleep(0.1)
            print('\r{:.1f}s'.format(time.time() - self.start_time), end = '')

    def start(self):
        if not self.end: return
        self.start_time = time.time()
        self.end = False
        self.thread = threading.Thread(target = self.loop)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        if self.end: return 0.0
        self.end = True
        self.thread.join()
        print('\r                \r', end = '')
        return time.time() - self.start_time
_timer = Timer()

def main():
    print('hello world, log')

if __name__ == '__main__':
    main()

