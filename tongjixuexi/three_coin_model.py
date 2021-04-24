import numpy as np
import math

class Three_coin:
    def __init__(self,pai=0.0,p=0.0,q=0.0):
        self.pai = pai
        self.p = p
        self.q = q

    def comput_y_sita(self,y):
        return self.pai*self.p**y*(1-self.p)**(1-y) + (1-self.pai)*self.q**y*(1-self.q)**(1-y)

    def log_P_Y_sita(self,X):
        result = 0
        for x in X:
            result += math.log(self.comput_y_sita(x))
        return result

    def compute_ui(self,y):
        return self.pai*self.p**y*(1-self.p)**(1-y)/self.comput_y_sita(y)

    def fit(self,X,max_iter):
        self.n = len(X)
        for i in range(max_iter):
            p_u1 = np.array([self.compute_ui(x)*x for x in X])
            ui = np.array([self.compute_ui(x) for x in X])
            q_ui = np.array([(1 - self.compute_ui(x))*x for x in X])
            self.pai = 1/self.n*sum(ui)
            self.p = sum(p_u1)/sum(ui)
            self.q = sum(q_ui)/sum(1-ui)

def main():
    X = [1,1,0,1,0,0,1,0,1,1]
    Three_coin_ = Three_coin(0.46,0.55,0.67)
    Three_coin_.fit(X,10)
    print(Three_coin_.pai,Three_coin_.p,Three_coin_.q)

if __name__ == '__main__':
    main()
