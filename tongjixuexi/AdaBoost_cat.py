import math
import numpy as np

class Adaboost_tree:
    def __init__(self,X,Y,feature_type='discrete'):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.N = len(X)
        self.feature_num = len(X[0])
        self.w = np.array([1/self.N] * self.N)
        self.g_x=[]
        self.feature_type=feature_type #特征类型
        self.get_feature_dict()


    def compute_error(self,y):
        y = np.array(y)
        return np.sum(self.w[y != self.Y])

    def compute_am(self,em):
        return 1/2*math.log((1-em)/em)

    def get_feature_dict(self):
        self.f_dict = {}
        for i in range(self.feature_num):
            self.f_dict[i] = list(set([x[i] for x in self.X]))

    def fit(self,max_iter=20):
        for iter in range(max_iter):
            index_list=[]
            error_list1=[]
            error_list2 = []
            pred_y_list1 = []
            pred_y_list2 = []

            if self.feature_type == 'discrete':
                for i in range(self.feature_num):
                    for j in self.f_dict[i]:
                        y1 = [1 if m[i] == j else -1 for m in self.X]
                        y2 = [-1 if m[i] == j else 1 for m in self.X]
                        error1 = self.compute_error(y1)
                        error2 = self.compute_error(y2)
                        index_list.append((i,j))
                        error_list1.append(error1)
                        error_list2.append(error2)
                        pred_y_list1.append(y1)
                        pred_y_list2.append(y2)

            if self.feature_type == 'continuous':
                for i in range(self.feature_num):
                    for j in self.f_dict[i]:
                        y1 = [1 if m[i] <= j else -1 for m in self.X]
                        y2 = [-1 if m[i] <= j else 1 for m in self.X]
                        error1 = self.compute_error(y1)
                        error2 = self.compute_error(y2)
                        index_list.append((i,j))
                        error_list1.append(error1)
                        error_list2.append(error2)
                        pred_y_list1.append(y1)
                        pred_y_list2.append(y2)

            if min(error_list1) <= min(error_list2):
                min_index = error_list1.index(min(error_list1))
                split_f_index,split_value = index_list[min_index]
                pred_y = pred_y_list1[min_index]
                positive = 1

            else:
                min_index = error_list2.index(min(error_list2))
                split_f_index,split_value = index_list[min_index]
                pred_y = pred_y_list2[min_index]
                positive = -1

            em = self.compute_error(pred_y)
            if em == 0:
                print('em is zero break')
                break
            am = self.compute_am(em)
            self.g_x.append((split_f_index,split_value,positive,am))
            w_list = self.w * np.exp(-am * self.Y * np.array(pred_y))
            self.w = w_list/np.sum(w_list)

    def predict_single(self,x):
        result = 0
        for split_f_index,split_value,positive,am in self.g_x:
            if self.feature_type == 'discrete':
                if x[split_f_index] == split_value:
                    result += positive * am
                else:
                    result += - positive * am
            elif self.feature_type == 'continuous':
                if x[split_f_index] <= split_value:
                    result += positive * am
                else:
                    result += - positive * am

        return np.sign(result)

    def predict(self,X):
        result = [self.predict_single(x) for x in X]
        return result

def main():
    X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3],
                  [0, 1, 2], [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])
    Y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])
    Adaboost_tree_ = Adaboost_tree(X,Y,feature_type='continuous')
    Adaboost_tree_.fit(20)
    print(Adaboost_tree_.predict(X))

if __name__ == '__main__':
    main()









