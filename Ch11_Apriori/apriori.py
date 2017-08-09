#!/usr/bin/env python
# encoding=utf8

import numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    dataSet为全部数据集
    返回元素个数为1的项集,如每个单独商品(去重后)列表
    '''
    C1 = [] #C1即为元素个数为1的项集,如每个单独商品列表
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    C1 = map(frozenset ,C1)  # map(frozenset, C1)的语义是将C1由Python列表转换为不变集合（frozenset，Python中的数据结构）。
    #frozenset是不可变的，用户不能修改。这里使用frozenset是因为要将这些集合作为字典键值使用，使用frozenset可以实现，而set却做不到。
    return C1

def scanD(D, Ck , minSupport):
    '''
    D为全部数据集
    Ck为大小为k（包含k个元素）的候选项集
    minSupport为设定的最小支持度
    
    返回值中retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度。
    '''
    ssCnt = {} # 记录每个项集出现次数
    
    #统计每个项集出现次数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt[can] + 1 if ssCnt.has_key(can) else 1
    
    numItems = float(len(D)) #全部数据集个数
    retList = []
    supportData = {}
    
    for key in ssCnt.iterkeys():
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key) # 将频繁项集插入返回列表的首部。
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk , k):
    '''
    该函数通过频繁项集列表Lk(实际是Lk-1)和项集个数k生成候选项集Lk。
    k的大小等于LK中频繁项集中的项数+1，eg: k == len(lk[0]) + 1
    
    注意其生成的过程中，首先对每个项集按元素排序，然后每次比较两个项集，只有在前k-1项相同时才将这两项合并。
    这样做是因为函数并非要两两合并各个集合，那样生成的集合并非都是k+1项的。在限制项数为k+1的前提下，只有在
    前k-1项相同、最后一项不相同的情况下合并才为所需要的新候选项集。

    由于Python中使用下标0表示第一个元素，因此代码中的[:k-2]的实际作用为取列表的前k-1个元素。
    
    '''
    retList = []
    lenLk = len(Lk)
    
    for i in range(lenLk):
        for j in range(i+1 , lenLk):
            # 若前k-2项相同时（也就是除了最后一项不相同，eg: abc与abe），将两个集合合并
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # frozenset集合并操作
    return retList
    
def apriori(dataSet, minSupport=0.5):
    '''
    Apriori算法的主函数。
    
    Ck表示项数为k的候选项集，最初的C1通过createC1()函数生成。Lk表示项数为k的频繁项集，supK为其支持度，Lk和supK由scanD()函数通过Ck计算而来。
    
    函数返回的L和supportData为所有的频繁项集及其支持度，因此在每次迭代中都要将所求得的Lk和supK添加到L和supportData中。
    
    '''
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    
    L1, supportData = scanD(D, C1, minSupport)
    
    L = [L1]
    k = 2
    
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supportDataK = scanD(D, Ck, minSupport)
        supportData.update(supportDataK)
        L.append(Lk)
        k +=1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    '''
    关联规则生成函数
    
    3个参数：频繁项集列表L、包含那些频繁项集支持数据的字典supportData、最小可信度阈值minConf
    
    '''
    bigRuleList = []  #包含可信度的规则列表,后面可以基于可信度对它们进行排序。
    
    for i in range(1, len(L)): #i = 0时L[0]为单个频繁集,i表示当前遍历的频繁项集包含的元素个数。这里只获取有两个或更多元素的集合
        for freqSet in L[i]:  # freqSet为当前遍历的频繁项集
            H1 = [frozenset([item]) for item in freqSet] # 对每个频繁项集构建只包含单个元素集合的列表H1
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf) #只有两个元素的集合
    return bigRuleList

def calcConf(freqSet, H , supportData, brl, minConf=0.7):
    '''
    对规则进行评估。计算规则的可信度，并过滤出满足最小可信度要求的规则，最后将这个规则列表添加到主函数的bigRuleList中（通过参数brl）。
    
    freqSet: 为有n个元素的频率集
    H：为频率集中的n个元素或n个元素的两两或多个组合
    supportData: 支持集
    brl: 用于存储>可信度minConf的关联规则
    minConf: 最低可信度
    
    返回值prunedH保存规则列表的右部
    
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq] #p(012->3) = p(0123 |012) = p(0123)/p(012)
        if conf >= minConf:
            print freqSet - conseq, ' ------> ', conseq, '   conf:', conf
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
        
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    根据当前候选规则集H生成下一层候选规则集
    
    参数：频繁项集freqSet，可以出现在规则右部的元素列表H，supportData保存项集的支持度，brl保存生成的关联规则，minConf同主函数
    
    '''
    m = len(H[0]) #计算H中的频繁项集大小m
    if len(freqSet) > (m + 1):  # 查看该频繁项集是否大到可以移除大小为m的子集
        Hmpl = aprioriGen(H, m+1) # 使用函数aprioriGen()来生成H中元素的无重复组合
        Hmpl = calcConf(freqSet, Hmpl, supportData, brl, minConf)
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)
    
    

if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    
    D = map(set, dataSet)
    
    L1, supportData0 = scanD(D, C1, 0.5)
    
    L, supportData = apriori(dataSet)
    
    rules = generateRules(L, supportData, minConf=0.5)
    
   