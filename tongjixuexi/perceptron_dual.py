import numpy as np
import matplotlib.pyplot as plt

class Perceptron_dual:
    def __init__(self,n=1,max_iter=10):
        self.max_iter = max_iter
        self.n  = n

    def fit(self,X,Y):
        X = np.array(X)
        Y = np.array(Y)
        sample_num = len(X)
        self.b = 0
        self.a = np.zeros(sample_num)
        self.w = np.sum((np.array(self.a) * np.array(Y)) * np.array(X).T, -1)
        #计算Gram矩阵
        self.G = np.zeros((sample_num,sample_num))
        for i in range(sample_num):
            for j in range(sample_num):
                self.G[i,j]=X[i].dot(X[j])
        self.iter = 0

        while bool(1-(self.predict(X) == Y).all()):
            for index,(i,j) in enumerate(zip(X,Y)):
                result = 0
                for m in range(sample_num):
                    result_mid = self.a[m]*Y[m]*self.G[m,index]
                    result += result_mid
                if j*(result + self.b) >0:
                    continue
                else:
                    self.a[index] += self.n
                    self.b += j*self.n
                    print(self.a,self.b)
            self.iter += 1
            if self.iter >= self.max_iter:
                print('cant not completely delieve the data')
                break
            self.w = np.sum((np.array(self.a) * np.array(Y)) * np.array(X).T, -1)
        print('training complete')
        pass

    def predict(self,X):
        return np.sign(np.sum(self.w * X, -1) + self.b)
def main():
    p1 = Perceptron_dual(1,20)
    x = [[3,3],[4,3],[1,1]]
    y = [1,1,-1]
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


