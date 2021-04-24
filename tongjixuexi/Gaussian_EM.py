import math
import numpy as np

class Gausian_EM:

    def __init__(self,Y,k):
        self.k = k
        self.Y = np.array(Y)
        self.feature_num = len(Y[0])
        self.N = len(Y)
        self.uk = []
        self.sik = []
        for i in range(k):
            self.uk.append(np.random.rand(self.feature_num))
            self.sik.append(np.random.rand(self.feature_num,self.feature_num))
        self.ak = np.array([1/k]*k)
        self.rjk = np.zeros((self.N,k)) + 0.001

    def caculate_y_sita(self,y,k_index):
        covdet = np.linalg.det(self.sik[k_index] + np.eye(self.feature_num) * 0.001)
        covinv = np.linalg.inv(self.sik[k_index] + np.eye(self.feature_num) * 0.001)
        denominator = ((2*math.pi)**self.feature_num * np.abs(covdet))**(1/2)
        numerator = np.exp(-0.5*((y-self.uk[k_index]).dot(covinv).dot(y-self.uk[k_index])))
        return numerator/denominator

    def compute_log_likelihood(self):
        result = 0
        for y in self.Y:
            result += np.log(np.array(np.sum([self.caculate_y_sita(y,k)*self.ak[k] for k in range(self.k)])))
        return result

    def fit(self,max_iter):
        for iter in range(max_iter):
            log_likelihood = self.compute_log_likelihood()
            for n in range(self.N):
                denominator = np.sum([self.ak[k] * self.caculate_y_sita(self.Y[n], k) for k in range(self.k)])
                for k_index in range(self.k):
                    self.rjk[n][k_index] = self.ak[k_index]*self.caculate_y_sita(self.Y[n], k_index)/denominator
            for k in range(self.k):
                self.ak[k] = np.sum([self.rjk[j][k] for j in range(self.N)])/float(self.N)
                self.sik[k] = np.sum([self.rjk[j][k]*((self.Y[j]-self.uk[k]).reshape(self.feature_num,1).dot((self.Y[j]-self.uk[k]).reshape(1,self.feature_num))) \
                                      for j in range(self.N)],axis=0)/np.sum([self.rjk[j][k] for j in range(self.N)])
                print([self.Y[j]*self.rjk[j][k] for j in range(self.N)])
                self.uk[k] = np.sum([self.Y[j]*self.rjk[j][k] for j in range(self.N)],axis=0)/np.sum([self.rjk[j][k] for j in range(self.N)])
            #
            # print('---------------------------')
            # print(self.ak)
            # print(self.sik)
            # print(self.uk)
            new_log_likelihood = self.compute_log_likelihood()
            if new_log_likelihood - log_likelihood < 0.0001:
                print('small fit')
                break

    def predict(self,x):
        return np.argmax([self.caculate_y_sita(x,k) for k in range(self.k)])

def main():
    X = []
    Y = []
    with open('../data/iris.data', 'r') as f:
        for i in f:
            data = i.split(',')
            X.append([float(j) for j in data[:4]])
            Y.append(data[4])
    Y = [1 if i == 'Iris-setosa\n' else 0 for i in Y]
    Gausian_EM_ = Gausian_EM(X,2)
    Gausian_EM_.fit(50)
    print([Gausian_EM_.predict(x) for x in X])
    print(Y)

if __name__ == '__main__':
    main()

