from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

class Eva:
    def __init__(self, y1, y2):
        self.mse = mean_squared_error(y1, y2)
        self.mae = mean_absolute_error(y1, y2)

    def __str__(self):
        return 'MSE={:.4f}, MAE={:.4f}'.format(self.mse, self.mae)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def mean(es):
        r = Eva.__new__(Eva)
        r.mse = np.mean([e.mse for e in es])
        r.mae = np.mean([e.mae for e in es])
        return r

    @staticmethod
    def better(a, b):
        if a.mse < b.mse:
            return a
        return b

    def is_better_than(self, other):
        if other is None:
            return True
        return self.mse < other.mse

    def prt(self):
        return '{}\t{}'.format(self.mse, self.mae)


def main():
    print('hello world, evaluate.py')


if __name__ == '__main__':
    main()
