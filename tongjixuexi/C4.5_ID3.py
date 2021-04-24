import numpy as np
import math
class Node:
    def __init__(self,feature_index=None,value=None,label=None):
        self.feature_index=feature_index
        self.value=value
        self.child=[]
        self.label=label


class C4_5:
    def __init__(self,X,Y,c=0.1,way='ID3'):
        self.c = c
        self.root=Node()
        self.X = X
        self.Y = Y
        self.feature_num = len(X[0])
        self.label_num = len(Y)
        self.feature_set = list(range(self.feature_num))
        self.getac()
        self.way = way

    def getac(self):
        self.dict_x = {}
        self.dict_y = set(self.Y)
        for i in range(self.feature_num):
            self.dict_x[i] = set([X[i] for X in self.X])

    @staticmethod
    def get_label(list_):
        return max(list_, key=list_.count)

    @staticmethod
    def count_Y(Y):
        dict_y = {}
        for i in Y:
            if i in dict_y.keys():
                dict_y[i]+=1
            else:
                dict_y[i] = 1
        return dict_y

    def experience_entropy(self,Y):
        dict_y = self.count_Y(Y)
        D = len(Y)
        set_y = set(Y)
        return -sum([dict_y[x]/D*math.log(dict_y[x]/D,2) for x in set_y])

    def get_feature(self,X,Y,rest_x):
        HD = self.experience_entropy(Y)
        Y = np.array(Y)
        X = np.array(X)
        entropy_ = []

        if self.way == 'ID3':
            for i in rest_x:
                sum_ = 0
                list_x = np.array([x[i] for x in X])
                for j in self.dict_x[i]:
                    sum__ = 0
                    Di = sum(list_x == j)
                    if Di != 0:
                        for m in self.dict_y:
                            Dik = sum(Y[list_x == j]==m)
                            if Dik != 0:
                                sum__ += Dik/Di*math.log(Dik/Di,2)
                    sum_ -= Di/len(list_x)*sum__
                add_entropy = HD - sum_
                entropy_.append(add_entropy)

        if self.way == 'C45':
            for i in rest_x:
                sum_ = 0
                list_x = np.array([x[i] for x in X])
                for j in self.dict_x[i]:
                    sum__ = 0
                    HAD = 0
                    Di = sum(list_x == j)
                    if Di != 0:
                        for m in self.dict_y:
                            Dik = sum(Y[list_x == j]==m)
                            if Dik != 0:
                                sum__ += Dik/Di*math.log(Dik/Di,2)
                    sum_ -= Di/len(list_x)*sum__
                    HAD -= Di/len(list_x)*math.log(Di/len(list_x),2)
                add_entropy = (HD - sum_)/HAD
                entropy_.append(add_entropy)

        max_add = max(entropy_)
        index_ = entropy_.index(max_add)
        spilt_feature = self.feature_set[index_]
        return spilt_feature,max_add


    def build_tree(self):
        def build_tree_(node, X, Y, rest_x):
            X = np.array(X)
            Y = np.array(Y)
            if len(set(Y))==1:
                node.label=list(set(Y))[0]
                return
            elif len(X[0])==0:
                node.label=self.get_label(self.Y)
                return
            spilt_feature,max_add = self.get_feature(X,Y,rest_x)
            if max_add < self.c:
                node.label=self.get_label(self.Y)
                return
            rest_x.remove(spilt_feature)
            for i in self.dict_x[spilt_feature]:
                Node_child = Node(feature_index=spilt_feature,value=i)
                build_tree_(Node_child,X[np.array([x[spilt_feature] for x in X])==i],Y[np.array([x[spilt_feature] for x in X])==i],rest_x)
                node.child.append(Node_child)
        build_tree_(self.root,self.X,self.Y,self.feature_set)

    def print_tree(self):
        root = self.root
        def pre_order(root):
            if root:
                print(root.feature_index,root.value,root.label,len(root.child))
            for i in root.child:
                pre_order(i)
        pre_order(root)

    def predict_single(self,X):
        if len(X) != self.feature_num:
            raise IndexError
        root = self.root
        while root.child:
            for node in root.child:
                if X[node.feature_index] == node.value:
                    root = node
                continue
        return root.value

    def predict(self,X):
        X = np.array(X)
        if len(X.shape) == 1:
            return self.predict_single(X)
        else:
            result = []
            for i in X:
                result.append(self.predict_single(i))
        return result



def main():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否']]
    X = [x[0:-1] for x in datasets]
    Y = [x[-1] for x in datasets]
    for i,j in zip(X,Y):
        print(i,j)
    C4_5_trainer = C4_5(X,Y)
    C4_5_trainer.build_tree()
    C4_5_trainer.print_tree()
    predict_single_x = [['中年', '是', '否', '一般'],['老年', '否', '否', '一般']]
    print(C4_5_trainer.predict((predict_single_x)))

if __name__ == '__main__':
    main()



