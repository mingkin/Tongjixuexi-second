import numpy as np

class SVD:
    def __init__(self,x):
        self.x = np.array(x)

    def get_r_rv(self,x):
        x = np.array(x)
        s_matrix = x.T.dot(x)
        r_list, r_v = np.linalg.eig(s_matrix)
        return r_list,r_v

    def get_matrix_rank(self):
        return np.linalg.matrix_rank(self.x)

    def get_V_M(self,x):
        r_list, r_v = self.get_r_rv(x)
        index_rank = np.argsort(-r_list)
        r_list = r_list[index_rank]
        r_list_ = r_list[r_list > 0]
        r_max = len(r_list_)
        V = r_v[:,index_rank]
        m = np.zeros((r_max,r_max))
        U = np.zeros((self.x.shape[0],r_max))
        for i in range(r_max):
            m[i][i] = np.sqrt(r_list_[i])
            U[:,i] = (np.dot(self.x, V[:, i]) / np.sqrt(r_list_[i])).T
        V = V[:, :r_max]
        return V,m,U

    def svd(self,way='norm',k=None):
        V,m,U= self.get_V_M(self.x)

        if way == 'norm':
            return V,m,U

        if way == 'truncated':
            r = self.get_matrix_rank()
            if k < r and k > 0:
                return V[:, :k], m[:k, :k], U[:, :k]
            else:
                return V, m, U

def main():
    x = np.array([[0,20,5,0,0],
                  [10,0,0,3,0],
                  [0,0,0,0,1],
                  [0,0,0,1,0]])
    svd = SVD(x)
    V, m, U = svd.svd()
    print(V)
    print(m)
    print(U)

    x = np.array([[0,20,5,0,0],
                  [10,0,0,3,0],
                  [0,0,0,0,1],
                  [0,0,0,1,0]])
    svd = SVD(x)
    V, m, U = svd.svd(way='truncated',k=2)
    print(V)
    print(m)
    print(U)

if __name__ == '__main__':
    main()

