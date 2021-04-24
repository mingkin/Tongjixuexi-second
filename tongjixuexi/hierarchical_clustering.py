import numpy as np
import copy
import matplotlib.pyplot as plt

class Hierarchical_cluster:

    def __init__(self,k=None,p=2,dis_way='min',c_way='agg'):
        self.k = k
        self.p = p
        self.dis_way = dis_way
        self.c_way = c_way

    def cauclate_dis(self, x1, x2):
        return np.sum(abs(x1 - x2) ** self.p) ** (1 / self.p)

    def create_D_matrix(self):
        self.D = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i==j:
                    self.D[i][j] = 0
                if i>j:
                    self.D[i][j] = self.cauclate_dis(self.x[i],self.x[j])
                    self.D[j][i] = self.cauclate_dis(self.x[i],self.x[j])

    def cauclate_cluster_dis(self,c1,c2):
        if self.dis_way == 'min':
            min_ = 999999999
            for i in c1:
                for j in c2:
                    min_ = min(min_,self.D[i][j])
            return min_

        if self.dis_way == 'mean':
            mean1 = np.mean([self.x[i] for i in c1],axis=0)
            mean2 = np.mean([self.x[i] for i in c2],axis=0)
            return self.cauclate_dis(mean1,mean2)

    def split_C(self,C):
        C1 = []
        C2 = []
        max_ = 0
        for i in C:
            for j in C:
                if j >= i:
                    if self.D[i][j] >= max_:
                        max_ = self.D[i][j]
                        max_i = i
                        max_j = j
        C1.append(max_i)
        C2.append(max_j)
        for c in C:
            if c == max_i or c == max_j:
                continue
            elif self.D[max_i][c] >= self.D[max_j][c]:
                C2.append(c)
            else:
                C1.append(c)
        return C1, C2

    def fit(self,x):
        self.x = np.array(x)
        self.n = len(x)
        self.create_D_matrix()

        if self.c_way == 'agg':
            C_way = []
            start_c = []
            for i in range(self.n):
                start_c.append([i])
            C_way.append(start_c)
            iter = 0
            while len(C_way[iter]) > 1:
                num_c = len(C_way[iter])
                c_temp = []
                dis_temp = []
                for c1 in range(num_c):
                    for c2 in range(num_c):
                        if c1 > c2:
                            c_temp.append([c1,c2])
                            dis_temp.append(self.cauclate_cluster_dis(C_way[iter][c1],C_way[iter][c2]))
                min_dis = min(dis_temp)
                min_index = dis_temp.index(min_dis)
                c1 = c_temp[min_index][0]
                c2 = c_temp[min_index][1]
                new_c = copy.deepcopy(C_way[iter])
                new_c.append(new_c[c1]+new_c[c2])
                del new_c[c1]
                del new_c[c2]
                C_way.append(new_c)
                iter += 1
            self.result = C_way


        if self.c_way == 'div':
            C_way = []
            new_c = list(range(self.n))
            C_way.append([new_c])
            iter = 0
            while len(C_way[iter]) < self.n:
                print(C_way)
                new_c = []
                for C in C_way[iter]:
                    if len(C) == 1 :
                        new_c.append(C)
                    else:
                        C1,C2 = self.split_C(C)
                        new_c.append(C1)
                        new_c.append(C2)
                C_way.append(new_c)
                iter+=1
            self.result = C_way

    def predict(self):
        final = np.zeros(len(self.x))
        if self.c_way == 'div':
            result  = self.result[self.k//2+1]
        if self.c_way == 'agg':
            result = self.result[-self.k]
        print(result)
        for i in range(len(result)):
            for j in result[i]:
                final[j] = i
        return list(final)

def main():
    x = [[0,2],[0,0],[1,0],[5,0],[5,2],[1,5],
         [2,4],[3,5],[1,1],[1,4],[5,4],[5,3],[0,1],
         [-1,0],[1,1],[4,4],[3,4]]
    k = 3
    HC = Hierarchical_cluster(k,dis_way='mean',c_way='agg')
    HC.fit(x)
    result = HC.predict()
    print(result)
    color_list =['r','k','b','g']
    for i,c in zip(range(len(result)),color_list[:len(result)]):
        positive_ = np.array(x)[np.array(result) == i]
        plt.scatter([k[0] for k in positive_],[k[1] for k in positive_] , c=c)
    plt.show()

if __name__ == '__main__':
    main()
