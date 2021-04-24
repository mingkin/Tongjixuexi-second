import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self,kernal='GKF',C=1):
        self.keranl=kernal
        self.b = 0
        self.X = None
        self.Y = None
        self.a = None
        self.N = None
        self.C = C
        self.feature_num = None
        self.Elist = None
        self.K = None

    def comput_kernal(self,x,y,sita=0.1):
        x = np.expand_dims(np.array(x),axis=0)
        y = np.expand_dims(np.array(y),axis=0)
        if self.keranl == 'GKF':
            return math.exp(-(x-y).dot((x-y).T)/2*sita**2)
        elif self.keranl == 'liner':
            return x.dot(y.T)[0][0]

    def compute_K(self):
        self.K = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.K[i][j] = self.comput_kernal(self.X[i],self.X[j])

    def conput_gx(self,index_x):
        gx = 0
        for j in range(self.N):
            gx += self.K[index_x][j] * self.Y[j] *self.a[j]
        gx += self.b
        return gx

    def compute_E_list(self):
        self.Elist = [self.conput_gx(i) - self.Y[i] for i in range(self.N)]

    def get_a1_index(self):
        for i in range(self.N):
            if self.a[i] < self.C and self.a[i] > 0:
                gi = self.conput_gx(i)
                if self.Y[i]*gi != 1:
                    return i

        for i in range(self.N):
            if self.a[i] == 0:
                if self.conput_gx(i)*self.Y[i] < 1:
                    return i
            elif self.a[i] == self.C:
                if self.conput_gx(i)*self.Y[i] > 1:
                    return i
        else:
            return -1

    def get_a2_index(self,a1_index):
        E1_E2 = [abs(self.Elist[a1_index] - self.Elist[i]) for i in range(self.N)]
        a2_index = E1_E2.index(max(E1_E2))
        return a2_index

    def fit(self,X,Y,max_iter=20):
        self.N = len(X)
        self.X = X
        self.Y = Y
        self.feature_num = len(X[0])
        self.a = np.zeros(self.N) + 0.5
        self.compute_K()

        for iter in range(max_iter):
            print('epoch = ' + str(iter))
            #更新E
            self.compute_E_list()
            #更新a1 a2
            a1_index = self.get_a1_index()
            if a1_index == -1:
                print('all_is_fit_KTT')
                break
            a2_index = self.get_a2_index(a1_index)
            a1_old = self.a[a1_index]
            a2_old = self.a[a2_index]
            L = max(0, a2_old + a1_old-self.C)
            H = min(self.C, a2_old + a1_old)
            n = self.K[a1_index][a1_index] + self.K[a2_index][a2_index] - 2*self.K[a1_index][a2_index]
            a2_new_unc = a2_old + self.Y[a2_index]*(self.Elist[a1_index]-self.Elist[a2_index])/n
            if a2_new_unc > H:
                a2_new = H
            elif a2_new_unc >= L and a2_new_unc <= H:
                a2_new = a2_new_unc
            elif a2_new_unc < L:
                a2_new = L
            a1_new = a1_old + self.Y[a1_index]*self.Y[a2_index]*(a2_old - a2_new)
            self.a[a1_index] = a1_new
            self.a[a2_index] = a2_new
            #更新b
            b1_new = - self.Elist[a1_index] - self.Y[a1_index]*self.K[a1_index][a1_index]*(a1_new-a1_old) \
            - self.Y[a2_index]*self.K[a2_index][a1_index]*(a2_new-a2_old) + self.b
            b2_new = - self.Elist[a2_index] - self.Y[a1_index]*self.K[a1_index][a2_index]*(a1_new-a1_old) \
            - self.Y[a2_index]*self.K[a2_index][a2_index]*(a2_new-a2_old) + self.b
            if 0 < a1_new < self.C and 0 < a2_new < self.C:
                self.b = (b1_new + b2_new) / 2
            elif 0 < a1_new < self.C:
                self.b = b1_new
            elif 0 < a2_new < self.C:
                self.b = b2_new

    def predict_single(self,x):
        result_1 = [self.a[i]*self.Y[i]*self.comput_kernal(x,self.X[i]) for i in range(self.N)]
        return np.sign(sum(result_1) + self.b)

    def predict(self,X):
        return [self.predict_single(x) for x in X]


def main():
    # X = [[1,2],
    #      [2,3],
    #      [3,3],
    #      [2,1],
    #      [3,2]]
    # Y = [1,1,1,-1,-1]
    X = []
    Y = []
    with open('../data/iris.data', 'r') as f:
        for i in f:
            data = i.split(',')
            X.append([float(j) for j in data[:4]])
            Y.append(data[4])
    Y = [1 if i == 'Iris-setosa\n' else -1 for i in Y]
    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        Y,
                                                        test_size=0.2,
                                                        random_state=9999)
    svm_trainer = SVM(C=30,kernal='GKF')
    svm_trainer.fit(train_X,train_y,max_iter=10)
    result = svm_trainer.predict(test_X)
    print(result)
    print(accuracy_score(test_y,result))

if __name__ == '__main__':
    main()











