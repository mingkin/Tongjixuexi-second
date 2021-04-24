import numpy as np

class MDP:
    def __init__(self,reward_list,p_statue_list,statue_num,action_num,gamma):
        self.reward_list = reward_list
        self.statue_num = statue_num
        self.v_pai = np.zeros(statue_num)
        self.action_num = action_num
        self.pai = np.zeros(action_num)
        self.gamma = gamma
        self.P_give_L_L = p_statue_list[0]
        self.P_give_L_H = p_statue_list[1]
        self.P_no_L_L = p_statue_list[2]
        self.P_no_L_H = p_statue_list[3]
        self.P_give_H_L = p_statue_list[4]
        self.P_give_H_H = p_statue_list[5]
        self.P_no_H_L = p_statue_list[6]
        self.P_no_H_H = p_statue_list[7]

    def update_value_pai(self,max_iter,e=None):
        for iter in range(max_iter):
            L_old = self.v_pai[0]
            H_old = self.v_pai[1]
            self.v_pai[0] = self.pai[0] * (self.reward_list[0] + self.P_give_L_L * L_old * self.gamma +
                                        self.P_give_L_H * H_old * self.gamma) + \
                            (1 - self.pai[0]) * (self.reward_list[1] + self.P_no_L_L * L_old * self.gamma +
                                        self.P_no_L_H * H_old * self.gamma)

            self.v_pai[1] = self.pai[1] * (self.reward_list[2] + self.P_give_H_L * L_old * self.gamma +
                                        self.P_give_H_H * H_old * self.gamma) + \
                            (1 - self.pai[1]) * (self.reward_list[3] + self.P_no_H_L * L_old * self.gamma +
                                        self.P_no_H_H * H_old * self.gamma)

            if e:
                if abs(self.v_pai[0] - L_old) < e and abs(self.v_pai[1] - H_old) < e:
                    break

    def update_pai(self):
        old_pai = self.pai.copy()
        L_give = self.reward_list[0] + self.P_give_L_L * self.v_pai[0] * self.gamma + self.P_give_L_H * self.v_pai[1] * self.gamma
        L_no = self.reward_list[1] + self.P_no_L_L * self.v_pai[0] * self.gamma + self.P_no_L_H * self.v_pai[1] * self.gamma
        if L_give > L_no:
            self.pai[0] = 1
        else:
            self.pai[0] = 0

        H_give = self.reward_list[2] + self.P_give_H_L * self.v_pai[0] * self.gamma + self.P_give_H_H * self.v_pai[1] * self.gamma
        H_no = self.reward_list[3] + self.P_no_H_L * self.v_pai[0] * self.gamma + self.P_no_H_H * self.v_pai[1] * self.gamma
        if H_give > H_no:
            self.pai[1] = 1
        else:
            self.pai[1] = 0

        if (old_pai == self.pai).all():
            return 1
        else:
            return None

    def fit(self,value_way='normal'):
        iter = 0
        while True:
            print(iter)
            iter+=1
            if value_way == 'normal':
                self.update_value_pai(100,1e-3)
            if value_way == 'value_iter':
                self.update_value_pai(1, 1e-3)
            result = self.update_pai()
            if result:
                break

def main():
    reward_list = np.array([5,10,35,25])
    statue_num = 2
    action_num = 2
    gamma = 0.9
    p_statue_list = [0.3,0.7,0.5,0.5,0.2,0.8,0.6,0.4]
    mdp = MDP(reward_list,p_statue_list,statue_num,action_num,gamma)
    mdp.fit(value_way='value_iter')
    print(mdp.v_pai)
    print(mdp.pai)

if __name__ == '__main__':
    main()








