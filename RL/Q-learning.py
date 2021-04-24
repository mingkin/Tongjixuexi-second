import numpy as np

class Flybrid:
    '''
    in the qusetion , i use Q-learing to slove this problem
    the update equation is self.Q[x,y,action] = self.Q[x,y,action] + self.a*(self.reward[move_end_x,move_end_y] +
                                     self.r * np.max(self.Q[move_end_x,move_end_y]) - self.Q[x,y,action])
    '''
    def __init__(self,a = 0.1, e = 0.3, r = 0.9 , number_move = 9, start = (3,0), G = (3,7)):
        self.a = a
        self.e = e
        self.r = r
        self.start_x,self.start_y = start
        self.G_x,self.G_y = G
        #build wind
        self.wind = np.zeros((7,10))
        self.wind[:,3:6] = 1
        self.wind[:,6:8] = 2
        self.wind[:,8] = 1
        print(self.wind)
        #build_reward
        self.reward = np.zeros((7, 10)) - 1
        self.reward[self.G_x,self.G_y] = 1
        print(self.reward)
        #build_Q_initital
        self.Q = np.zeros((7, 10, number_move))
        '''
        0  = go up
        1  = go down
        2 =  go right
        3 =  go left
        4 =  go up_right
        5 =  go down_right
        6 =  go up_left
        7 =  go down_left
        8 =  stay
        '''
        self.number_move = number_move
        move_list = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(0,0)]
        self.move_list = move_list[:number_move]
        self.row = self.wind.shape[0]
        self.col = self.wind.shape[1]

    def converge(self,local):
        x,y = local
        if x > self.row - 1:
            x = self.row - 1
        if x < 0:
            x = 0
        if y > self.col -1:
            y = self.col -1
        if y < 0:
            y = 0
        return x, y

    def move(self,x,y,move):
        move_no_wind = (x + self.move_list[move][0],y + self.move_list[move][1])
        new_x,new_y = self.converge(move_no_wind)
        move_result = (new_x+self.wind[new_x,new_y],new_y)
        result_x, result_y = self.converge(move_result)
        return int(result_x),int(result_y)

    def greedy_action(self,x,y):
        p_list = np.ones(self.number_move) * self.e/(self.number_move-1)
        p_list[np.argmax(self.Q[x,y])] = 1 - self.e
        take_action = np.random.choice(list(range(self.number_move)),p=p_list)
        return take_action

    def iter(self,max_iter=10000):
        for iter in range(max_iter):
            x, y = self.start_x, self.start_y
            step = 0
            while (x,y) != (self.G_x,self.G_y):
                action = self.greedy_action(x,y)
                move_end_x, move_end_y = self.move(x,y,action)
                self.Q[x,y,action] = self.Q[x,y,action] + self.a*(self.reward[move_end_x,move_end_y] +
                                     self.r * np.max(self.Q[move_end_x,move_end_y]) - self.Q[x,y,action])
                x,y = move_end_x,move_end_y
                step+=1

    def get_result(self):
        best_q = np.argmax(self.Q,-1)

        print()
        for i in range(self.row-1,-1,-1):
            print(best_q[i])

        tract = np.zeros((self.row,self.col))
        x, y = self.start_x, self.start_y
        step = 0
        while (x,y) != (self.G_x,self.G_y):
            tract[x,y] = 1
            x,y = self.move(x,y,best_q[x,y])
            step += 1
        tract[x, y] = 1
        print()
        print('min_step = ', step)
        for i in range(self.row-1,-1,-1):
            print(tract[i])

def main():
    '''
    number_move = 4 : 动作集合：← ↑ → ↓
    number_move = 8 : 动作集合：← ↑ → ↓ ↖ ↗ ↘ ↙
    number_move = 9 : 动作集合：← ↑ → ↓ ↖ ↗ ↘ ↙ + stop
    a               : 学习率
    e               : e-greedy
    r               : 折损率
    start           : 起点
    G               : 终点
    return: 各个位置的最优动作图 、 最小步长 、 行动轨迹图（1表示）
    '''
    fly = Flybrid(a = 0.1, e = 0.3, r = 0.9 , number_move = 9, start = (3,0), G = (3,7))
    fly.iter()
    fly.get_result()

if __name__ == '__main__':
    main()
