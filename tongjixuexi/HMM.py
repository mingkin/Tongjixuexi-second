import numpy as np

class HMM:
    def __init__(self,A=None,B=None,Pi=None,O = None):
        if A:
            self.A = np.array(A)
        else:
            self.A = None
        if Pi:
            self.Pi = np.array(Pi)
            self.i_num = len(Pi)
        else:
            self.Pi = None
            self.i_num = None
        if B:
            self.B = np.array(B)
            self.o_num = np.array(B).shape[-1]
        else:
            self.B = None
            self.o_num = None
        self.O = O
        if O:
            self.T = len(O)
        else:
            self.T = None
        self.ati = None
        self.bti = None

    def update_ati(self):
        self.ati = np.zeros((self.T,self.i_num))
        for i in range(self.i_num):
            self.ati[0][i] = self.Pi[i]*self.B[i][self.O[0]]
        for t in range(self.T-1):
            for i in range(self.i_num):
                self.ati[t+1][i] = np.sum(self.ati[t]*self.A[:,i])*self.B[i][self.O[t+1]]

    def update_bti(self):
        self.bti = np.zeros((self.T,self.i_num))
        for i in range(self.i_num):
            self.bti[self.T-1][i] = 1
        for t in range(self.T-2,-1,-1):
            for i in range(self.i_num):
                self.bti[t][i] = np.sum(self.A[i,:]*self.B[:,self.O[t+1]]*self.bti[t+1])

    def comput_ri(self, t, i):
        return self.ati[t,i]*self.bti[t,i]/np.sum(self.ati[t]*self.bti[t])

    def comput_ei(self, t, i, j):
        return self.ati[t,i]*self.A[i,j]*self.B[j][self.O[t+1]]*self.bti[t+1][j] /\
    np.sum(self.ati[t]*self.A*self.B[:,self.O[t+1]]*self.bti[t+1])

    def predict_whole_p_o(self):
        self.update_ati()
        return np.sum(self.ati[self.T-1])

    def predict_q_give_i_t(self,t,i):
        self.update_ati()
        self.update_bti()
        return self.comput_ri(t-1,i-1)

    def predict_p_t_t_1_q(self,t,i,j):
        self.update_ati()
        self.update_bti()
        return self.comput_ei(t-1,i-1,j-1)

#---------------------------------------------
    def Baum_welch(self,O,i_num,max_iter=20):
        self.T = len(O)
        self.O = np.array(O)
        self.i_num = i_num
        self.A = np.zeros((i_num,i_num))+ 1/i_num
        self.B = np.zeros((i_num,len(set(O)))) + 1/len(set(O))
        self.Pi = np.zeros(i_num) + 1/i_num
        for n in range(max_iter):
            self.update_ati()
            self.update_bti()
            for i in range(i_num):
                self.Pi[i] = self.comput_ri(0,i)
                for j in range(i_num):
                    sum_et = 0
                    sum_rt = 0
                    for t in range(self.T-1):
                        sum_et += self.comput_ei(t,i,j)
                        sum_rt += self.comput_ri(t,i)
                    self.A[i][j] = sum_et/sum_rt

            for j in range(i_num):
                for k in range(len(set(O))):
                    sum_rt1 = 0
                    sum_rt2 = 0
                    for t in range(self.T):
                        rtj = self.comput_ri(t,j)
                        if self.O[t] == k:
                            sum_rt1 += rtj
                        sum_rt2 += rtj
                    self.B[j][k] = sum_rt1/sum_rt2

    def Viterbi(self):
        p_marix = np.zeros((self.T,self.i_num))
        rout = [[0] for i in range(self.i_num)]
        for i in range(self.i_num):
            p_marix[0][i] = self.Pi[i]*self.B[i][self.O[0]]
        for t in range(1,self.T):
            for i in range(self.i_num):
                max_p = 0
                max_rout = 0
                index_rout = 0
                for j in range(self.i_num):
                    max_p = max(max_p,p_marix[t-1][j]*self.A[j][i]*self.B[i][self.O[t]])
                    if p_marix[t-1][j]*self.A[j][i] > max_rout:
                        max_rout = p_marix[t-1][j]*self.A[j][i]
                        index_rout = j
                p_marix[t][i] = max_p
                rout[i].append(index_rout)
        P_max = max(p_marix[self.T-1])
        max_index = np.argmax(p_marix[self.T-1])
        rout[max_index].append(max_index)
        return P_max , [x+1 for x in rout[max_index][1:]]

def main():
    A = [[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]
    B = [[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    Pi = [0.2,0.4,0.4]
    O = [0,1,0,1,1,0,1,0,0]
    HMM_test = HMM(A=A,B=B,Pi=Pi,O=O)
    print(HMM_test.predict_whole_p_o())
    print(HMM_test.predict_q_give_i_t(4,3))
    print(HMM_test.predict_p_t_t_1_q(3, 1, 2))
    print(HMM_test.ati)
    print(HMM_test.bti)
    P_max, rout = HMM_test.Viterbi()
    print(P_max, rout)

    HMM_test_predict = HMM()
    HMM_test_predict.Baum_welch(O=O,i_num=3,max_iter=50)
    P_max, rout = HMM_test_predict.Viterbi()
    print(P_max, rout)

if __name__ == '__main__':
    main()







