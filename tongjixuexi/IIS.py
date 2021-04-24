import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.metrics import accuracy_score

def load_data(path):
    def transform(row):
        return list((row[1:] > 128).apply(int))
    data = pd.read_csv(path,header=None)
    x_list = list(data.apply(transform,axis=1))
    y_list = list(data[0])
    return x_list,y_list

class Maximum_Entropy:

    def __init__(self,X,Y, way='IIS'):
        self.X = X
        self.Y = Y
        self.feature_num = np.array(X).shape[-1]
        self.M = 10000
        self.sample_num = len(X)
        self.f_num = 0
        self.label_list = list(set(Y))
        self.label_num = len(set(Y))
        self.w = None
        self.create_fixy()
        self.create_index()
        self.EP_fi()
        self.way = way


    def create_fixy(self):
        self.fi_dict = [defaultdict(int) for i in range(self.feature_num)]
        for i in range(self.feature_num):
            for j in range(self.sample_num):
                x = self.X[j][i]
                y = self.Y[j]
                self.fi_dict[i][(x,y)] += 1

    def create_index(self):
        self.fixy_id = [{} for i in range(self.feature_num)]
        self.id_xy = {}
        index = 0
        for i in range(self.feature_num):
            for x,y in self.fi_dict[i]:
                self.fixy_id[i][(x,y)] = index
                self.id_xy[index] = (x,y)
                index += 1
        self.f_num = index
        self.w = [0] * self.f_num

    def EP_fi(self):
        self.EP_fi_list = [0] * self.f_num
        for i in range(self.feature_num):
            for x, y in self.fi_dict[i]:
                self.EP_fi_list[self.fixy_id[i][(x,y)]] = self.fi_dict[i][(x, y)]/self.sample_num

    def EPfi(self):
        self.EPfi_list = [0] * self.f_num
        for i in range(self.sample_num):
            x = self.X[i]
            for j in self.label_list:
                P_Y_X = self.caculate_P_Y_X(x,j)
                for m in range(self.feature_num):
                    if (x[m],j) in self.fixy_id[m]:
                        index = self.fixy_id[m][(x[m],j)]
                        self.EPfi_list[index] += P_Y_X * 1/self.sample_num

    def caculate_P_Y_X(self,X,y):
        fenzi = 0
        fenmu = 0
        for j in self.label_list:
            sum_1 = 0
            for i in range(self.feature_num):
                x = X[i]
                if (x, j) in self.fi_dict[i]:
                    index = self.fixy_id[i][(x, j)]
                    sum_1 += self.w[index]
            if j == y:
                fenzi = math.exp(sum_1)
            fenmu += math.exp(sum_1)
        return fenzi/fenmu

    def updata_w(self,max_iter):
        for i in range(max_iter):
            print(i)
            self.EPfi()
            self.w += [1/self.M * math.log(self.EP_fi_list[i]/self.EPfi_list[i]) for i in range(self.f_num)]

#-----------DFP-------------------------------

    def compute_gk(self):
        self.EPfi()
        return np.array(self.EPfi_list) - np.array(self.EP_fi_list)

    def compute_fw(self):
        left_part = 0
        for i in range(self.sample_num):
            fenmu = 0
            x_ = self.X[i]
            for j in self.label_list:
                sum_1 = 0
                for i in range(self.feature_num):
                    x = x_[i]
                    if (x, j) in self.fi_dict[i]:
                        index = self.fixy_id[i][(x, j)]
                        sum_1 += self.w[index]
                fenmu += math.exp(sum_1)
            left_part += math.log(fenmu) * 1/self.sample_num
        right_part = np.sum(np.array(self.EP_fi_list) * np.array(self.w))
        return left_part - right_part

    def DFP(self,eplison=1,max_iter=20):
        self.GK = np.array(np.eye(self.f_num),dtype=float)
        gk = self.compute_gk()
        if np.linalg.norm(gk,ord=2) < eplison:
            print(self.w)
            return

        for i in range(max_iter):
            pk = -self.GK.dot(gk)
            temp_w = self.w
            fw_list = []
            rate = 0.005
            for k in range(20):
            #线性搜索20次
                # print('i is'+str(i)+'k is'+str(k))
                self.w = pk * rate * k + temp_w
                fw = self.compute_fw()
                fw_list.append(fw)

            min_index = fw_list.index(min(fw_list))
            print('iter is '+str(i)+' min_index is '+ str(min_index))
            deta_w = pk * rate * min_index
            self.w = deta_w + temp_w
            gk_ = self.compute_gk()

            if np.linalg.norm(gk_,ord=2) < eplison:
                print(self.w)
                break

            yk = (gk_ - gk).reshape(self.f_num,1)
            si_k = deta_w.reshape(self.f_num,1)
            gk = gk_
            PK = si_k.dot(si_k.T) / si_k.T.dot(yk)
            QK = self.GK.dot(yk).dot(yk.T).dot(self.GK)/yk.T.dot(self.GK).dot(yk)
            self.GK = self.GK + PK -QK

#-------------------------------------

    def model_fit(self,max_iter):
        if self.way == 'IIS':
            self.updata_w(max_iter=max_iter)
        if self.way == 'DFP':
            self.DFP(max_iter=max_iter)

    def predict(self, X):
        result = []
        for x in X:
            result_x = []
            for y in self.label_list:
                p_y = self.caculate_P_Y_X(x, y)
                result_x.append(p_y)
            result.append(result_x)
        result_y = np.argmax(result, -1)
        return result_y

def main():
    path_train = '../data/mnist_train.csv'
    path_test = '../data/mnist_test.csv'
    x_train,y_train = load_data(path_train)
    x_test, y_test = load_data(path_test)
    Maximum_Entropy_ = Maximum_Entropy(x_train[:10000],y_train[:10000],way='DFP')
    Maximum_Entropy_.model_fit(10)
    y_pred = Maximum_Entropy_.predict(x_test[:10])
    print(accuracy_score(y_test[:10], y_pred))

if __name__ == '__main__':
    main()









