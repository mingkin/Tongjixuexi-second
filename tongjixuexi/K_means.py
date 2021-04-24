import numpy as np
import matplotlib.pyplot as plt

class K_menas:
    def __init__(self,k=None,p=2):
        self.k = k
        self.p = p

    def cauclate_dis(self, x1, x2):
        return np.sum(abs(x1 - x2) ** self.p) ** (1 / self.p)

    def fit(self,x):
        self.x = np.array(x)
        self.n = len(x)
        initial_c = np.random.choice(list(range(self.n)),self.k,replace=False)
        self.mean_list = [self.x[i] for i in initial_c]

        while True:
            self.c_list = [[] for _ in range(self.k)]
            self.x_list = [[] for _ in range(self.k)]
            for i in range(self.n):
                min_dis = 9999999
                for j in range(self.k):
                    dis = self.cauclate_dis(self.x[i], self.mean_list[j])
                    if min_dis > dis:
                        min_dis = dis
                        c = j
                self.c_list[c].append(i)
                self.x_list[c].append(self.x[i])
            mean_list_ = np.array([np.mean(c,0) for c in self.x_list])
            if (mean_list_ == self.mean_list).all():
                break
            self.mean_list = mean_list_

    def predict(self):
        dict_ = {}
        for i in range(len(self.c_list)):
            for j in self.c_list[i]:
                dict_[j]=i
        return [dict_[i] for i in range(self.n)]

    def compute_d(self,c):
        max_ = -1
        for c1 in range(len(c)):
            for c2 in range(len(c)):
                if c1 >= c2:
                    max_ = max(max_,self.cauclate_dis(c[c1],c[c2]))
        return max_

    def compute_mean_d(self):
        return np.mean([self.compute_d(i) for i in self.x_list])

def main():
    x = [[0,2],[0,0],[1,0],[5,0],[5,2],[1,5],
         [2,4],[3,5],[1,1],[1,4],[5,4],[5,3],[0,1],
         [-1,0],[1,1],[4,4],[3,4]]
    k_ = 5
    KM = K_menas(k_)
    KM.fit(x)
    result = KM.predict()
    print(result)

    k_list=[]
    for k in [1,2,3,4,5,6,7]:
        KM = K_menas(k)
        KM.fit(x)
        k_list.append([k,KM.compute_mean_d()])
    print(k_list)

    color_list =['r','k','b','g']
    for i,c in zip(range(len(result)),color_list[:len(result)]):
        positive_ = np.array(x)[np.array(result) == i]
        plt.scatter([k[0] for k in positive_],[k[1] for k in positive_] , c=c)
    plt.show()

if __name__ == '__main__':
    main()


