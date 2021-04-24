import numpy as np
import collections
import jieba

class PLSA:
    def __init__(self,text_list,k):
        self.k = k
        self.text_list = text_list
        self.text_num = len(text_list)
        self.get_X()

    def get_X(self):
        self.cuted_text = [jieba.lcut(text,cut_all=True) for text in self.text_list]
        self.word_all = []
        for i in self.cuted_text:
            self.word_all.extend(i)
        self.word_set = list(set(self.word_all))
        self.word_num = len(self.word_set)
        self.word_dict = {}
        for index,word in enumerate(self.word_set):
            self.word_dict[word] = index
        self.X = np.zeros((self.word_num,self.text_num))
        for i in range(self.text_num):
            count_ = collections.Counter(self.cuted_text[i])
            for k, v in count_.items():
                self.X[self.word_dict[k],i] = v

    def update_p_z_wd(self):
        self.z_wd  = np.zeros((self.word_num,self.text_num,self.k))
        for i in range(self.word_num):
            for j in range(self.text_num):
                self.z_wd[i,j] = np.array([self.w_z[i]*self.z_d[:,j]]) / np.sum([self.w_z[i]*self.z_d[:,j]])


    def fit(self,max_iter):
        self.w_z  = np.random.random((self.word_num,self.k))
        self.z_d = np.random.random((self.k,self.text_num))

        for iter in range(max_iter):
            self.update_p_z_wd()
            for k in range(self.k):
                for i in range(self.word_num):
                    self.w_z[i,k] = np.sum(self.X[i]*self.z_wd[i,:,k])/\
                    np.sum(self.X*self.z_wd[:,:,k])
                for j in range(self.text_num):
                    self.z_d[k,j] = np.sum(self.X[:,j]*self.z_wd[:,j,k])/np.sum(self.X[:,j])


def main():
    text_list = [
    '一个月前，足协杯十六进八的比赛，辽足费尽周折对调主客场，目的只是为了葫芦岛体育场的启用仪式。那场球辽足5比0痛宰“主力休息”的天津泰达。几天后中超联赛辽足客场对天津，轮到辽足“全替补”，\
    1比3输球，甘为天津泰达保级的祭品。那时，辽足以“联赛保级问题不大，足协杯拼一拼”作为主力和外援联赛全部缺阵的理由。',
    '被一脚踹进“忘恩负义”坑里的孙杨，刚刚爬出来，又有手伸出来，要把孙杨再往坑里推。即使是陪伴孙杨参加世锦赛的张亚东(微博)教练，\
    也没敢大义凛然地伸出援手，“孙杨愿意回去我不拦”，球又踢给了孙杨。张亚东教练怕什么呢？',
    '孙杨成绩的利益分配，以及荣誉的分享，圈里人都知道，拿了世界冠军和全运冠军，运动员都会有相应的高额奖金，那么主管教练也会得到与之对应的丰厚奖励，\
    所以孙杨获得的荣誉，也会惠及主管教练。']
    lsa = PLSA(text_list,k=2)
    lsa.fit(10)
    print(lsa.w_z)
    print(lsa.z_d)

if __name__ == '__main__':
    main()




