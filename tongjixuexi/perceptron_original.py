import numpy as np
import matplotlib.pyplot as plt

class Perceptron_orginal:
    def __init__(self,n=1,max_iter=10):
        self.rate = n
        self.max_iter = max_iter

    def fit(self,X,Y):
        X = np.array(X)
        Y = np.array(Y)
        self.feature_num = len(X[0])
        self.w = np.zeros(self.feature_num)
        self.b = 0
        self.iter = 0
        while bool(1-(self.predict(X) == Y).all()):
            for i,j in zip(X,Y):
                result = self.predict(i)*j
                if result >0:
                    continue
                else:
                    self.w += i*j*self.rate
                    self.b += j*self.rate
            self.iter += 1
            if self.iter >= self.max_iter:
                print('cant not completely delieve the data')
                break
        print('training complete')
        pass

    def predict(self,X):
        return np.sign(np.sum(self.w * X, -1) + self.b)
def main():
    p1 = Perceptron_orginal(0.5,20)
    x = [[3,3],[4,3],[1,1],[1,2],[6,2]]
    y = [1,1,-1,1,-1]
    p1.fit(x,y)
    print(p1.predict(x))
    print(p1.w)
    print(p1.b)

    positive_ =  np.array(x)[np.array(y) == 1]
    negetive_ = np.array(x)[np.array(y) == -1]
    plt.scatter([k[0] for k in positive_],[k[1] for k in positive_],c='r',label='1')
    plt.scatter([k[0] for k in negetive_], [k[1] for k in negetive_],c='b',label='0')
    x_ = np.arange(0, 10, 0.1)
    y_ = -(p1.w[0] * x_ + p1.b) / p1.w[1]
    plt.plot(x_, y_)
    plt.show()

if __name__ == '__main__':
    main()


