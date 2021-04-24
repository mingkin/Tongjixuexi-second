import numpy as np
import jieba

class LDA:
    def __init__(self,text_list,k):
        self.k = k
        self.text_list = text_list
        self.text_num = len(text_list)
        self.get_X()
        self.NKV = np.zeros((self.k,self.word_num))
        self.NMK = np.zeros((self.text_num,self.k))
        self.nm = np.zeros(self.text_num)
        self.nk = np.zeros(self.k)
        self.zmn = [[] for i in range(self.text_num)]
        self.alpha = np.random.randint(1,self.k,size=k)
        self.beta = np.random.randint(1,self.word_num, size=self.word_num)


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

    def initial_K(self):
        for doc_num in range(self.text_num):
            for word in self.cuted_text[doc_num]:
                k = np.random.choice(self.k, 1)[0]
                self.zmn[doc_num].append(k)
                v = self.word_dict[word]
                self.NMK[doc_num,k] += 1
                self.nm[doc_num] += 1
                self.NKV[k,v] += 1
                self.nk[k] += 1

    def iter_jbs(self):
        for doc_num in range(self.text_num):
            for word_index in range(len(self.cuted_text[doc_num])):
                v = self.word_dict[self.cuted_text[doc_num][word_index]]
                k = self.zmn[doc_num][word_index]
                self.NMK[doc_num,k] -= 1
                self.nm[doc_num] -= 1
                self.NKV[k,v] -= 1
                self.nk[k] -= 1
                p_klist = (self.NKV[:,v]+self.beta[v])/np.sum(self.NKV[:,v]+self.beta[v])*(self.NMK[doc_num]+self.alpha[k])/np.sum(self.NMK[doc_num]+self.alpha[k])
                p_klist = p_klist/np.sum(p_klist)
                k_choice = np.random.choice(self.k,p = p_klist)
                self.zmn[doc_num][word_index] = k_choice

                self.NMK[doc_num,k_choice] += 1
                self.nm[doc_num] += 1
                self.NKV[k_choice,v] += 1
                self.nk[k_choice] += 1

    def get_sita_y(self):
        self.sita_mk  = np.zeros((self.text_num,self.k))
        self.yta_kv = np.zeros((self.k,self.word_num))
        for i in range(self.text_num):
            self.sita_mk[i] = (self.NMK[i]+self.alpha)/np.sum(self.NMK[i])
        for j in range(self.k):
            self.yta_kv[j] = (self.NKV[j]+self.beta)/np.sum(self.NKV[j])

    def fit(self,max_iter = 100):
        self.initial_K()
        for iter in range(max_iter):
            print(iter)
            self.iter_jbs()
        self.get_sita_y()

def main():
    text_list = [
    '一个月前，足协杯十六进八的比赛，辽足费尽周折对调主客场，目的只是为了葫芦岛体育场的启用仪式。那场球辽足5比0痛宰“主力休息”的天津泰达。几天后中超联赛辽足客场对天津，轮到辽足“全替补”，\
    1比3输球，甘为天津泰达保级的祭品。那时，辽足以“联赛保级问题不大，足协杯拼一拼”作为主力和外援联赛全部缺阵的理由。',
    '被一脚踹进“忘恩负义”坑里的孙杨，刚刚爬出来，又有手伸出来，要把孙杨再往坑里推。即使是陪伴孙杨参加世锦赛的张亚东(微博)教练，\
    也没敢大义凛然地伸出援手，“孙杨愿意回去我不拦”，球又踢给了孙杨。张亚东教练怕什么呢？',
    '孙杨成绩的利益分配，以及荣誉的分享，圈里人都知道，拿了世界冠军和全运冠军，运动员都会有相应的高额奖金，那么主管教练也会得到与之对应的丰厚奖励，\
    所以孙杨获得的荣誉，也会惠及主管教练。']
    k = 2
    lda = LDA(text_list,k)
    lda.fit()
    print(lda.sita_mk)
    print(lda.yta_kv)

if __name__ == '__main__':
    main()













