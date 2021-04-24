import numpy as np
class PCA:
    def __init__(self,x,R=None):
        self.x = np.array(x)
        self.dim = self.x.shape[-1]
        self.num = self.x.shape[0]
        self.R = R
        self.X = None
        self.r_list = None
        self.r_v = None
        self.nk = None
        self.X_ = None

    def standetlize(self):
        self.X = []
        for i in range(self.dim):
            mean = np.mean(self.x[:,i])
            sii = 1/(self.num-1) * np.sum((self.x[:,i] - mean)**2)
            self.X.append((self.x[:, i]-mean)/sii)
        self.X = np.transpose(self.X)

    def get_R(self):
        self.R = 1/(self.num-1)*(self.X.T.dot(self.X))

    def get_r_rv(self,x):
        x = np.array(x)
        s_matrix = x.T.dot(x)
        r_list, r_v = np.linalg.eig(s_matrix)
        index_rank = np.argsort(-r_list)
        r_list = r_list[index_rank]
        r_v = r_v.T[index_rank]
        return r_list,r_v

    def fit(self,k=None,sup=None,way='R'):
        self.standetlize()
        if way == 'R':
            self.get_R()
            r_list,r_v = self.get_r_rv(self.R)
            nk = r_list/np.sum(r_list)
            if sup:
                for i in range(len(nk)):
                    if np.sum(nk[:i+1]) > sup:
                        k = i+1
                        break
            if k:
                r_v = r_v[:k]
                r_list = r_list[:k]
                self.r_v = r_v
                self.r_list = r_list
                self.nk = nk[:k]
                y = []
                for i in range(k):
                    y.append(r_v[i].dot(self.X.T))
                y = np.transpose(y)

        if way == 'SVD':
            self.X_ = 1/np.sqrt(self.num-1)*self.X
            u,s,v = np.linalg.svd(self.X_)
            y = v[:,:k].T.dot(self.X.T)
            y = np.transpose(y)
        return y


    def get_rv(self):
        return self.r_v

    def get_r_list(self):
        return self.r_list

    def get_nk(self):
        return self.nk

    def get_factor_loading(self):
        return [np.sqrt(self.r_list[i])*(self.r_v[i]) for i in range(len(self.r_list))]

    def get_vi(self):
        return np.sum([(np.sqrt(self.r_list[i]) * (self.r_v[i]))**2 for i in range(len(self.r_list))],0)

def main():
    x = [[2,9,2],
         [3,4,5],
         [3,5,2],
         [4,5,9],
         [5,6,1],
         [7,8,0]]
    pca = PCA(x)
    print(pca.fit(sup=0.7))
    print(pca.get_factor_loading())
    print(pca.get_vi())
    print(pca.fit(k=2,way="SVD"))

if __name__ == '__main__':
    main()





