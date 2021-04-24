import numpy as np

class FM_reg:
    def __init__(self, k=None, alpha=None, max_iter=None, way='GD', batch_size=None, norm=False):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.way = way
        self.batch_size = batch_size
        self.norm = norm

    def normalization_train(self):
        for m in range(self.feature_num):
            _range = np.max(self.x[:, m]) - np.min(self.x[:, m])
            self.x[:, m] = (self.x[:, m] - np.min(self.x[:, m])) / _range
            self.x_test[:, m] = (self.x_test[:, m] - np.min(self.x[:, m])) / _range

    def fit(self, x, y, x_test, y_test):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.feature_num = len(x[0])
        self.wo = 0
        self.w = np.random.random(self.feature_num)
        self.v = np.random.random((self.feature_num,self.k))
        if self.norm:
            self.normalization_train()

        if self.way == 'GD':
            for i in range(self.max_iter):
                _wo = 0
                _w = np.zeros(self.feature_num)
                _v = np.zeros((self.feature_num, self.k))
                for n in range(len(self.x)):
                    y_ = self.compute_y(self.x[n])
                    loss = -(self.y[n] - y_) * self.alpha
                    _wo -= loss * 1
                    _w -= loss * self.x[n]
                    for m in range(self.feature_num):
                        _v[m] -= loss * (self.x[n][m] * (self.x[n].dot(self.v)) - (self.x[n][m] ** 2) * self.v[m])
                self.wo += _wo / len(self.x)
                self.w += _w / len(self.x)
                self.v += _v / len(self.x)
                print('train', i, self.compute_mae(self.x, self.y))
                print('test', i, self.compute_mae(self.x_test, self.y_test))

        if self.way == 'SGD':
            for i in range(self.max_iter):
                for n in range(len(self.x)):
                    y_ = self.compute_y(self.x[n])
                    loss = -(self.y[n] - y_) * self.alpha
                    _wo -= loss * 1
                    _w -= loss * self.x[n]
                    for m in range(self.feature_num):
                        _v[m] -= loss * (self.x[n][m] * (self.x[n].dot(self.v)) - (self.x[n][m] ** 2) * self.v[m])
                    self.wo += _wo
                    self.w += _w
                    self.v += _v
                print('train', i, self.compute_mae(self.x, self.y))
                print('test', i, self.compute_mae(self.x_test, self.y_test))

        if self.way == 'BGD':
            for i in range(self.max_iter):
                shuffle_ix = np.random.permutation(np.arange(len(self.x)))
                self.x = self.x[shuffle_ix]
                self.y = self.y[shuffle_ix]
                _wo = 0
                _w = np.zeros(self.feature_num)
                _v = np.zeros((self.feature_num, self.k))
                for n in range(len(self.x)):
                    y_ = self.compute_y(self.x[n])
                    loss = -(self.y[n] - y_) * self.alpha
                    _wo -= loss * 1
                    _w -= loss * self.x[n]
                    for m in range(self.feature_num):
                        _v[m] -= loss * (self.x[n][m] * (self.x[n].dot(self.v)) - (self.x[n][m] ** 2) * self.v[m])
                    if (n + 1) % self.batch_size == 0:
                        self.wo += _wo / self.batch_size
                        self.w += _w / self.batch_size
                        self.v += _v / self.batch_size
                        _wo = 0
                        _w = np.zeros(self.feature_num)
                        _v = np.zeros((self.feature_num, self.k))
                print('train', i, self.compute_mae(self.x, self.y))
                print('test', i, self.compute_mae(self.x_test, self.y_test))

    def compute_mae(self, x_, y_):
        return np.mean(np.abs(y_ - np.array([self.compute_y(x) for x in x_])))

    def compute_y(self, x):
        y = self.wo + np.sum(x * self.w) + 1 / 2 * np.sum(x.dot(self.v) ** 2 - (x ** 2).dot(self.v ** 2))
        return y