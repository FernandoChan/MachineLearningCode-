dataset_num = 201630588238 % 349
print(dataset_num) # 63

algr_str = 'LMNBCTRAGP'
maps = {'L':'线性判别分析LDA',
            'M':'支持向量机',
           'N':'最近邻分类器',
           'B':'朴素贝叶斯；',
           'C':'决策树C4.5',
           'T':'分类与回归树',
         'R':'随机森林',
        'A':'Adaboost',
        'G':'Gradient Tree Boosting',
        'P':'标签传播',}
name = 'CXW'

def judge(name, algr):
    name_list  = list_2ord(list(name.lower()))
    algr_list = list_2ord(list(algr.lower()))

    if len(name_list) == 2:
        name_list.append((name_list[-1]+1)%26)
    elif len(name_list)> 3:
        name_list = name_list[:3]
    if len(algr_list)<=3 :
        return algr

    res = []
    if name_list.__len__()==3:
        for i in range(algr_list.__len__()-name_list.__len__()+1):
            sum = 0
            for j in range(name_list.__len__()):
                sum += pow(abs(algr_list[i+j]-name_list[j]),2)
            res.append(sum)
        inx  = res.index(max(res))
        return algr[inx : inx+3]
import logging
def list_2ord (obj):
    try:
        if(isinstance(obj,list)):
            res = [ord(i) - 96 for i in obj if ord(i) < 122 and ord(i) >= 97]
            return res
        elif(isinstance(obj,str)):
            return list_2ord(list(obj.lower()))
        else:
            raise TypeError('invalid sequence %s'%obj)
    except TypeError as e:
        logging.exception(e)
        raise
print(judge(name,algr_str))
print([maps[i] for i in judge(name,algr_str)])

# output: ['随机森林', 'Adaboost', 'Gradient Tree Boosting']

print()