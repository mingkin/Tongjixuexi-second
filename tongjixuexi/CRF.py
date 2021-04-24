import numpy as np

class CRF:
    def __init__(self,y=None,x=None,y_num=None,x_num=None,N=None):
        self.y = y
        self.x = x
        self.y_num = y_num
        self.x_num = x_num
        self.N = N
        self.get_feature()
        self.build_Marix(self.x[0])

    def get_feature(self):
        self.ti = [lambda y_1, y, x, i: 1 if i == 2 and y_1 == 1 and y == 2 else 0,
              lambda y_1, y, x, i: 1 if i == 3 and y_1 == 1 and y == 2 else 0,
              lambda y_1, y, x, i: 1 if i == 2 and y_1 == 1 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 3 and y_1 == 2 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 2 and y_1 == 2 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 3 and y_1 == 2 and y == 2 else 0,
         ]
        self.w_ti = [1,1,0.6,1,1,0.2]
        self.si = [lambda y_1, y, x, i: 1 if i == 1 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 1 and y == 2 else 0,
              lambda y_1, y, x, i: 1 if i == 2 and y == 2 else 0,
              lambda y_1, y, x, i: 1 if i == 2 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 3 and y == 1 else 0,
              lambda y_1, y, x, i: 1 if i == 3 and y == 2 else 0,
         ]
        self.w_si = [1,0.5,0.5,0.8,0.8,0.5]
        self.fk = self.ti+self.si
        self.wk = self.w_ti+self.w_si

    def build_Marix(self,x):
        self.Marix = np.zeros((self.N+1,self.y_num,self.y_num))
        for i in range(self.N+1):
            for n in range(self.y_num):
                if i == self.N:
                    self.Marix[i][:,0] = 1
                    break
                for m in range(self.y_num):
                    for k in range(len(self.fk)):
                        if i == 0:
                            if n==0:
                                self.Marix[i][0][m] += self.wk[k] * (self.fk[k](0,m+1,x[i],i+1))
                        else:
                            self.Marix[i][n][m] += self.wk[k] * (self.fk[k](n+1,m+1,x[i],i+1))
        self.Marix = np.exp(self.Marix)

    def get_aiT(self):
        self.aiT = np.zeros((self.N+1,self.y_num))
        self.aiT[0,:] = 1
        for i in range(1,self.N+1):
            self.aiT[i] = self.aiT[i-1].dot(self.Marix[i])

    def get_biT(self):
        self.biT = np.zeros((self.N + 1, self.y_num))
        self.biT[self.N,:] = 1
        for i in range(self.N-1,-1,-1):
            self.biT[i] = self.Marix[i+1].dot(self.biT[i+1])

#--------概率计算问题---------------
    def predict_Py_i(self,i,yi):
        self.get_aiT()
        self.get_biT()
        return self.aiT[i][yi-1]*self.biT[i][yi-1]/np.sum(self.aiT[self.N])

    def compute_p(self,y):
        result = 1
        for i in range(self.N):
            if i == 0:
                Z = self.Marix[i]
                result *= self.Marix[i][0][y[i]-1]
            else:
                Z =  Z.dot(self.Marix[i])
                result *= self.Marix[i][y[i-1]-1][y[i]-1]
        return result/np.sum(Z)

#----------维特比预测算法-------------------------
    def Viterbi(self,x):
        p_marix = np.zeros((len(x),self.y_num))
        rout = [[0] for i in range(self.y_num)]
        for j in range(self.y_num):
            p_marix[0][j] = np.sum(np.array([f(0,j+1,x[0],1) for f in self.fk])*self.wk)
        for i in range(1,len(x)):
            for l in range(self.y_num):
                max = 0
                rout_tem = 0
                for j in range(self.y_num):
                    wf = p_marix[i - 1][j] + np.sum(np.array([f(j+1, l+1, x[i], i + 1) for f in self.fk]) * self.wk)
                    if wf > max:
                        max = wf
                        rout_tem = j
                p_marix[i][l] = max
                rout[l].append(rout_tem)
        max_result = np.max(p_marix[len(x)-1])
        max_index = list(p_marix[len(x)-1]).index(np.max(p_marix[len(x)-1]))
        rout[max_index].append(max_index)
        return max_result,[i+1 for i in rout[max_index][1:]]

# -----------学习算法--------------------------------
    def compute_pxy_f(self):
        #pfk
        list_pxyti = np.zeros(len(self.ti))
        list_pxysi = np.zeros(len(self.si))
        #p_fk
        list_countti = np.zeros(len(self.ti))
        list_countsi = np.zeros(len(self.si))
        for x,y in zip(self.x,self.y):
            self.build_Marix(x)
            self.get_aiT()
            self.get_biT()
            for i in range(len(x)):
                list_temp = np.zeros(len(self.ti))
                for k in range(len(self.ti)):
                    if i == 0:
                        right = 1*self.Marix[0][0][y[i]-1]*self.biT[0][y[i]-1]
                        left = self.ti[k](i,y[i],x[i],i+1)
                        list_temp[k] += left*right/np.sum(self.aiT[self.N])
                    else:
                        right = self.aiT[i][y[i-1]-1] * self.Marix[i][y[i-1]-1][y[i]-1]*self.biT[i][y[i]-1]
                        left = self.ti[k](y[i-1],y[i],x[i],i+1)
                        list_temp[k] += left*right/np.sum(self.aiT[self.N])
                    if left == 1:
                        list_countti[k] += 1
                list_pxyti += list_temp

            for i in range(len(x)):
                list_temp = np.zeros(len(self.si))
                for k in range(len(self.si)):
                    right = self.aiT[i][y[i]-1]*self.biT[i][y[i]-1]
                    left = self.si[k](y[i-1], y[i], x[i], i+1)
                    list_temp[k] += left * right / np.sum(self.aiT[self.N])
                    if left == 1:
                        list_countsi[k] += 1
                list_pxysi += list_temp
        return list_countti/len(self.x),list_countsi/len(self.x),list_pxyti/len(self.x),list_pxysi/len(self.x)

    def compute_fw(self):
        left = 0
        right = 0
        for x in self.x:
            self.build_Marix(x)
            self.get_aiT()
            left += np.log(np.sum(self.aiT[self.N]))

        for x, y in zip(self.x, self.y):
            for i in range(len(x)):
                for k in range(len(self.fk)):
                    if i == 0:
                        right += self.wk[k] * self.fk[k](0,y[i],x[i],i+1)
                    else:
                        right += self.wk[k] * self.fk[k](y[i-1], y[i], x[i], i+1)
        return (left - right)/len(self.x)

    def fit(self,max_iter=3,how='IIS',lr=0.001):
        self.w_ti = [0]*len(self.ti)
        self.w_si = [0]*len(self.si)
        self.wk = self.w_ti + self.w_si
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        if how == 'IIS':
            S = 20
            for i in range(max_iter):
                ep_tk,ep_sk,eptk,epsk = self.compute_pxy_f()
                if np.linalg.norm(1/S*np.log(ep_tk/eptk), ord=2)+np.linalg.norm(1/S*np.log(ep_sk/epsk),ord=2) < 0.1:
                    print('when iter is '+str(i)+' shoulian')
                    break
                self.w_ti += 1/S*np.log(ep_tk/eptk)
                self.w_si += 1/S*np.log(ep_sk/epsk)
                self.wk = list(self.w_ti) + list(self.w_si)

        #按着最大熵模型的公式写的，用EP - E_P为倒数，不知道对不对
        elif how == 'GD':
            for i in range(max_iter):
                ep_tk,ep_sk,eptk,epsk = self.compute_pxy_f()
                gt = eptk - ep_tk
                gs = epsk - ep_sk
                fanshut = np.linalg.norm(gt,ord=2)
                fanshus = np.linalg.norm(gs,ord=2)
                if fanshut + fanshus < 0.75:
                    print('when iter is '+str(i)+' shoulian')
                    break
                temp_w_ti = self.w_ti
                temp_w_si = self.w_si
                fw_list = []
                #线性搜索这里有问题，fw会一直缩小，因该是fw的计算出现了错误
                for k in range(20):
                    self.w_ti = temp_w_ti - lr * gt * k
                    self.w_si = temp_w_si - lr * gs * k
                    self.wk = list(self.w_ti) + list(self.w_si)
                    fw = self.compute_fw()
                    fw_list.append(fw)
                min_index = fw_list.index(min(fw_list))
                deta_wti = gt * lr * min_index
                deta_wsi = gs * lr * min_index
                self.w_ti = temp_w_ti - deta_wti
                self.w_si = temp_w_si - deta_wsi
                self.wk = list(self.w_ti) + list(self.w_si)

def main():
    y = [[1,2,2],[2,1,1],[1,1,1],[1,2,2],[2,2,2],[2,1,2],[1,1,2],[1,2,1],[1,2,2],[1,2,2],[1,2,2],[1,2,2]]
    x = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
    CRF_test = CRF(y=y,y_num=2,N=3,x=x)
    print(CRF_test.Marix)
    print(CRF_test.compute_p(y[0]))
    print(CRF_test.predict_Py_i(2,1))
    print(CRF_test.Viterbi(x[0]))
    CRF_test.fit(50, how='IIS')
    print(CRF_test.wk)
    CRF_test.fit(500,how='GD',lr=0.001)
    print(CRF_test.wk)

if __name__ == '__main__':
    main()