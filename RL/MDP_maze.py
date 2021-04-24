import numpy as np

class maze_mdp:
    def __init__(self,maze,gamma):
        self.maze = maze
        self.row = maze.shape[0]
        self.col = maze.shape[1]
        self.r_s = np.array([x for y in maze for x in y])
        self.end = np.where(self.r_s > 0)[0][0]
        self.trap = np.where(self.r_s < -1)[0]
        self.v_s = np.zeros(len(self.r_s))
        self.maze_number = np.arange(0, len(self.r_s)).reshape(maze.shape)
        # 0 : up 85%
        # 1 : down 85%
        # 2 : left 85%
        # 3 : right 85%
        self.p1 = 0.85
        self.p2 = 0.05
        self.action = np.zeros(len(self.r_s))
        self.gamma = gamma

    def go_up(self,x):
        if 0 <= x-self.col:
            return x-self.col
        else:
            return x

    def go_left(self,x):
        if x in  self.maze_number[:,0]:
            return x
        else:
            return x-1

    def go_right(self,x):
        if x in  self.maze_number[:,-1]:
            return x
        else:
            return x+1

    def go_down(self,x):
        if x+self.col <= self.end:
            return x+self.col
        else:
            return x

    def go(self,x):
        return self.go_up(x),self.go_down(x),self.go_left(x),self.go_right(x)

    def update_value_pai(self):
        old_v_s = self.v_s.copy()
        for i in range(len(self.r_s)):
            if i == self.end or i in self.trap:
                self.v_s[i] = 0
                continue
            up , down , left ,right  = self.go(i)

            if self.action[i] == 0:
                self.v_s[i] = self.p1 *(self.r_s[up] +  self.gamma *old_v_s[up]) + \
                              self.p2 *(self.r_s[down] + self.gamma *old_v_s[down]) + \
                              self.p2 *(self.r_s[left] + self.gamma *old_v_s[left]) + \
                              self.p2 *(self.r_s[right] + self.gamma *old_v_s[right])

            if self.action[i] == 1:
                self.v_s[i] = self.p1 *(self.r_s[down] + self.gamma*old_v_s[down]) + \
                              self.p2 *(self.r_s[up] +  self.gamma*old_v_s[up]) + \
                              self.p2 *(self.r_s[left] + self.gamma * old_v_s[left]) + \
                              self.p2 *(self.r_s[right] + self.gamma * old_v_s[right])

            if self.action[i] == 2:
                self.v_s[i] = self.p1 *(self.r_s[left] + self.gamma*old_v_s[left]) + \
                              self.p2 *(self.r_s[up] + self.gamma*old_v_s[up]) + \
                              self.p2 *(self.r_s[down] + self.gamma * old_v_s[down]) + \
                              self.p2 *(self.r_s[right] + self.gamma * old_v_s[right])

            if self.action[i] == 3:
                self.v_s[i] = self.p1 *(self.r_s[right] + self.gamma*old_v_s[right]) + \
                              self.p2 *(self.r_s[up] + self.gamma*old_v_s[up]) + \
                              self.p2 *(self.r_s[left] + self.gamma * old_v_s[left]) + \
                              self.p2 *(self.r_s[down] + self.gamma * old_v_s[down])

    def update_pai(self):
        action = self.action.copy()
        for i in range(len(self.action)):
            up, down, left, right = self.go(i)
            up_v = self.p1 *(self.r_s[up] + self.gamma * self.v_s[up]) + \
                              self.p2 *(self.r_s[down] + self.gamma * self.v_s[down]) + \
                              self.p2 *(self.r_s[left] + self.gamma * self.v_s[left]) + \
                              self.p2 *(self.r_s[right] + self.gamma * self.v_s[right])

            down_v = self.p1 * (self.r_s[down] + self.gamma * self.v_s[down]) + \
                          self.p2 * (self.r_s[up] + self.gamma * self.v_s[up]) + \
                          self.p2 * (self.r_s[left] + self.gamma * self.v_s[left]) + \
                          self.p2 * (self.r_s[right] + self.gamma * self.v_s[right])

            left_v = self.p1 * (self.r_s[left] + self.gamma * self.v_s[left]) + \
                          self.p2 * (self.r_s[up] + self.gamma * self.v_s[up]) + \
                          self.p2 * (self.r_s[down] + self.gamma * self.v_s[down]) + \
                          self.p2 * (self.r_s[right] + self.gamma * self.v_s[right])

            right_v = self.p1 * (self.r_s[right] + self.gamma * self.v_s[right]) + \
                          self.p2 * (self.r_s[up] + self.gamma * self.v_s[up]) + \
                          self.p2 * (self.r_s[left] + self.gamma * self.v_s[left]) + \
                          self.p2 * (self.r_s[down] + self.gamma * self.v_s[down])

            v_list = [up_v,down_v,left_v,right_v]
            if i == 0:
                print(v_list)

            self.action[i] = v_list.index(max(v_list))
        if (action == self.action).all():
            return 1
        else:
            return None

    def fit(self):
        iter = 0
        while True:
            print(iter)
            iter += 1
            self.update_value_pai()
            result = self.update_pai()
            print(self.action)
            if result:
                break

def main():
    maze = np.array([[-100,-100, -100, -100,-100, -100],
                     [-100,-1, -1,   -1,   -1,    -100],
                     [-100,-1, -100, -100, -100,  -100],
                     [-100,-1, -100, -1,   -1,    -100],
                     [-100,-1,  -1,  -1,    -1,   -100],
                     [-100, -1, -100,  -1,    -1,   -100],
                     [-100, -100,-1, -100,  -1,   -100],
                     [-100, -1,  -1,   -1,   100,   -100],
                     [-100,-100, -100, -100,-100, -100]])
    mdp =  maze_mdp(maze,0.8)
    mdp.fit()
    print(mdp.v_s.reshape((9,6)))
    print(mdp.action.reshape((9,6)))

if __name__ == '__main__':
    main()

