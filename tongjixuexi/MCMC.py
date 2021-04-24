import numpy as np
import matplotlib.pyplot as plt

class MCMC:
    def __init__(self,p=None):
        self.p = p
        self.x1 = np.random.random(1)[0]
        self.x2 = np.random.random(1)[0]

    def update_x1(self):
        self.x1 = np.random.normal(loc=self.p*self.x2, scale=np.sqrt(1-self.p**2), size=1)[0]

    def update_x2(self):
        self.x2 = np.random.normal(loc=self.p*self.x1, scale=np.sqrt(1-self.p**2), size=1)[0]

    def fit(self,n,m):
        self.sample_list = []
        self.x1_list = []
        self.x2_list = []
        for i in range(m):
            self.update_x1()
            self.update_x2()
            if i > n :
                self.sample_list.append((self.x1,self.x2))
                self.x1_list.append(self.x1)
                self.x2_list.append(self.x2)

    def plot(self):
        plt.hist(self.x1_list,bins=50,alpha=0.3)
        plt.hist(self.x2_list,bins=50,alpha=0.3)
        plt.hist(np.random.normal(loc=0, scale=np.sqrt(1-self.p**2), size=5000),bins=50,alpha=.3)
        plt.show()

def main():
    mc = MCMC(0.5)
    mc.fit(5000,10000)
    print(mc.sample_list)
    mc.plot()

if __name__ == '__main__':
    main()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta




class MCMC:
    def __init__(self,scale=0.5):
        self.ta = np.random.random(1)
        self.scale = 0.5

    def update_ta(self):
        ta_n = np.random.normal(loc=self.ta, scale=self.scale, size=1)[0]
        a = min(1,beta.pdf(ta_n,1,1)/beta.pdf(self.ta,1,1))
        u = np.random.random(1)
        if u <= a :
            self.ta = ta_n

    def fit(self,n,m):
        self.sample_list = []
        for i in range(m):
            self.update_ta()
            if i > n :
                self.sample_list.append(self.ta)

    def plot(self):
        plt.hist(self.sample_list,bins=50,alpha=0.3)
        plt.show()

def main():
    mc = MCMC(0.2)
    mc.fit(5000,10000)
    print(np.mean([beta.pdf(x,1,1) for x in mc.sample_list])*0.4)
    mc.plot()

if __name__ == '__main__':
    main()


# '''
# Created on 2018年5月16日
# p:输入的概率分布，离散情况采用元素为概率值的数组表示
# N:认为迭代N次马尔可夫链收敛
# Nlmax:马尔可夫链收敛后又取的服从p分布的样本数
# isMH:是否采用MH算法，默认为True
#
# 1）输入我们任意选定的马尔科夫链状态转移矩阵Q，平稳分布π(x)，设定状态转移次数阈值n1，需要的样本个数n2
# 2）从任意简单概率分布采样得到初始状态值x0
# 3）for t=0 to n1+n2−1:
# 　a) 从条件概率分布Q(x|xt)中采样得到样本x∗
# 　b) 从均匀分布采样u∼uniform[0,1]
# 　c) 如果u<α(xt,x∗)=π(x∗)Q(x∗,xt), 则接受转移xt→x∗，即xt+1=x∗
# 　d) 否则不接受转移，即xt+1=xt
# 样本集(xn1,xn1+1,...,xn1+n2−1)即为我们需要的平稳分布对应的样本集。
# '''
#
# from __future__ import division
# import matplotlib.pyplot as plt
# import numpy as np
# from array import array
#
# def mcmc(Pi ,Q,N=1000,Nlmax=100000,isMH=False):
#     X0 = np.random.randint(len(Pi))# 第一步：从均匀分布（随便什么分布都可以）采样得到初始状态值x0
#     T = N+Nlmax-1
#     result = [0 for i in range(T)]
#     t = 0
#     while t < T-1:
#         t = t + 1
#         # 从条件概率分布Q(x|xt)中采样得到样本x∗
#         # 该步骤是模拟采样，根据多项分布，模拟走到了下一个状态
#         #（也可以将该步转换成一个按多项分布比例的均匀分布来采样）
#         x_cur = np.argmax(np.random.multinomial(1,Q[result[t-1]]))  # 第二步：取下一个状态 ，采样候选样本
#         if isMH:
#             '''
#                 细致平稳条件公式：πi Pij=πj Pji,∀i,j
#             '''
#             a = (Pi[x_cur] * Q[x_cur][result[t-1]]) /(Pi[result[t-1]] * Q[result[t-1]][x_cur])  # 第三步：计算接受率
#             acc = min(a ,1)
#         else: #mcmc
#             acc = Pi[x_cur] * Q[x_cur][result[t-1]]
#         u = np.random.uniform(0 ,1)  # 第四步：生成阈值
#         if u< acc:  # 第五步：是否接受样本
#             result[t]=x_cur
#         else:
#             result[t]= result[t-1]
#     return result
#
# def count(q, n):
#     L = array("d")
#     l1 = array("d")
#     l2 = array("d")
#     for e in q:
#         L.append(e)
#     for e in range(n):
#         l1.append(L.count(e))
#     for e in l1:
#         l2.append(e / sum(l1))
#     return l1, l2
#
# if __name__ == '__main__':
#     Pi = np.array([0.5, 0.2, 0.3]) # 目标的概率分布
#     #状态转移矩阵，但是不满足在 平衡状态时和 Pi相符
#     #我们的目标是按照某种条件改造Q ，使其在平衡状态时和Pi相符
#     #改造方法就是，构造矩阵 P，且 P(i,j)=Q(i,j)α(i,j)
#     #                          α(i, j) = π(j)Q(j, i)
#     #                          α(j, i) = π(i)Q(i, j)
#     Q = np.array([[0.9, 0.075, 0.025],
#                   [0.15, 0.8, 0.05],
#                   [0.25, 0.25, 0.5]])
#
#     a = mcmc(Pi,Q)
#     l1 = ['state%d' % x for x in range(len(Pi))]
#     plt.pie(count(a, len(Pi))[0], labels=l1, labeldistance=0.3, autopct='%1.2f%%')
#     plt.title("markov:" +str(Pi))
#     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
#
# np.random.seed(42)
# x = np.linspace(-20, 20, 100)
# # raw_y=stats.expon(scale=1).pdf(x)
# raw_y = stats.norm.pdf(x, 0, 5)
#
# Samp_Num = 10000
# result = []
# init = 1
# result.append(init)
# # p=lambda r:stats.expon(scale=1).pdf(r)
# p = lambda r: stats.norm.pdf(r, 0, 5)
# q = lambda v: stats.norm.rvs(loc=v, scale=2, size=1)
#
# for i in range(Samp_Num):
#     xstar = q(result[i])
#     alpha = min(1, p(xstar) / p(result[i]))
#     u = np.random.rand(1)
#     if u < alpha:
#         result.append(xstar)
#     else:
#         result.append(result[i])
#     print(i)
#
# n, bins, patches = plt.hist(result, 50, density=1, facecolor='blue', alpha=0.5)
# plt.plot(x, raw_y)
# plt.show()
