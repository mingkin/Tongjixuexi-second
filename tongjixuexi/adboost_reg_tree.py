from tongjixuexi import reg_tree
import numpy as np

class adboost_reg_tree:
    def __init__(self):
        self.tree_list = []

    def fit(self,X,Y,max_iter=5,min_leave_data=3):
        self.X = np.array(X)
        self.Y = np.array(Y)
        for i in range(max_iter):
            reg_t = reg_tree.Cart_reg(self.X, self.Y, min_leave_data)
            reg_t.build_tree()
            pred_y = np.array(reg_t.predict(self.X))
            print(pred_y)
            self.tree_list.append(reg_t)
            self.Y = self.Y - pred_y
            if (self.Y == 0).all():
                print('total_fit')
                break

    def predict(self,X):
        result = np.zeros(len(X))
        for i in self.tree_list:
            y = i.predict(X)
            result += np.array(y)
        return result

def main():
    X=[[1,5,7,4,8,1,2],
       [2,3,5,5,2,7,8],
       [1,2,3,4,5,6,7],
       [1,2,1,2,2,3,9],
       [2,8,9,7,0,1,4],
       [4,8,3,4,5,6,7],
       [4,1,3,1,5,8,0]]
    Y= [2,6,2,5,8,3,2]
    adboost_reg_tree_ = adboost_reg_tree()
    adboost_reg_tree_.fit(X,Y,max_iter=5,min_leave_data=4)
    print(adboost_reg_tree_.predict(X))

if __name__ == '__main__':
    main()
