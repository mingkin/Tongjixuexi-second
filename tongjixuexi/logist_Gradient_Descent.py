import numpy as np
import matplotlib.pyplot as plt

class logist:
    def __init__(self,a=1,c=None,max_iter=9999):
        self.w = None
        self.a = a
        self.c = c
        self.max_iter = max_iter

    def sigmoid(self,x):
        x = np.array(x)
        return np.exp(x.dot(self.w))/(1+np.exp(x.dot(self.w)))

    def expand_b_for_x(self,x):
        x = np.array(x)
        if len(x.shape)==1:
            x = np.concatenate((x,np.array([1])))
        else:
            x = np.concatenate((x,np.ones(len(x)).reshape(len(x), 1)),axis=1)
        return x

    def fit(self,x,y):
        x = self.expand_b_for_x(x)
        self.w = np.zeros(len(x[0]))
        grade = [999,999,999]
        iter = 0
        while (np.abs(grade) * self.a).any() > self.c and iter <= self.max_iter:
            grade = np.sum(x * (y - self.sigmoid(x)).reshape(len(y), 1), axis=0)
            deta_w = grade * self.a
            self.w += deta_w
            iter += 1
        return

    def predict(self,x):
        x = self.expand_b_for_x(x)
        return self.sigmoid(x)

def main():
    x = [[3,3],[4,3],[1,1],[1,2],[6,2]]
    y = [1,1,0,1,0]
    logist_classer = logist(a=1,c=0.6,max_iter=20)
    logist_classer.fit(x,y)
    print(logist_classer.w)
    print(logist_classer.predict(x))
    positive_ =  np.array(x)[np.array(y) == 1]
    negetive_ = np.array(x)[np.array(y) == 0]
    plt.scatter([k[0] for k in positive_],[k[1] for k in positive_],c='r',label='1')
    plt.scatter([k[0] for k in negetive_], [k[1] for k in negetive_],c='b',label='0')
    x_ = np.arange(0, 10, 0.1)
    y_ = -(logist_classer.w[0] * x_ + logist_classer.w[-1]) / logist_classer.w[1]
    plt.plot(x_, y_)
    plt.show()

if __name__ == '__main__':
    main()