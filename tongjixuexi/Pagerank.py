import numpy as np

class PageRank:
    def __init__(self,M,D=0.85):
        self.M = np.array(M)
        self.D = D
        self.n = self.M.shape[0]

    def iter_way(self,max_iter=100,e=1e-3):
        self.R = np.ones(self.n)/self.n
        dm = self.M * self.D
        smooth = (1-self.D)/self.n*np.ones(self.n)
        for iter in range(max_iter):
            old_R =  self.R
            self.R = dm.dot(self.R) + smooth
            if np.sum(np.abs(self.R - old_R)) < e:
                print("change_small")
                break

    def power_way(self,max_iter=100,e=1e-3):
        self.X = np.ones(self.n)/self.n
        self.A = self.D*self.M + np.ones((self.n,self.n))*(1-self.D)/self.n
        for iter in range(max_iter):
            old_x = self.X
            self.X = self.norm(self.A.dot(self.X))
            if np.sum(np.abs(self.X - old_x)) < e:
                print("change_small")
                break

        for col in range(self.n):
            self.X = self.X / np.sum(self.X)

        self.R = self.X

    def math_way(self):
        self.R = np.linalg.inv(np.identity(self.n) - self.D*self.M).dot((1-self.D)/self.n*np.ones(self.n))

    @staticmethod
    def norm(y):
        return y/np.max(y)

    def fit(self,max_iter=100,e=1e-3,way='power'):
        if way == 'power':
            self.power_way(max_iter,e)

        if way == 'iter':
            self.iter_way(max_iter,e)

        if way == 'math':
            self.math_way()

def main():
    M = np.array([[0,0,1],
                  [1/2,0,0],
                  [1/2,1,0]])
    pageR = PageRank(M)
    pageR.fit(max_iter=100,e=1e-3,way='power')
    print(pageR.R)
    pageR.fit(max_iter=100,e=1e-3,way='iter')
    print(pageR.R)
    pageR.fit(max_iter=100,e=1e-3,way='math')
    print(pageR.R)

if __name__ == '__main__':
    main()