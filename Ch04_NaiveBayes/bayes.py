#! /usr/bin/env python
# enconding=utf8

import numpy as np

def loadDataSet():
    '''
    训练数据集与类别标签
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  #1 代表侮辱性文字 ， 0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    '''
    根据训练数据集生成词汇表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #集合合并，集合不同于list，会自动删除相同的元素。
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    将文本转化成词向量
    vocabList:词汇表
    inputSet: 某文本
    '''
    returnVec = [0]*len(vocabList) #初始化一个所有元素都是0的，有词汇表等长的词向量。0表示未出现对应词汇
    
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s not in vocabulary list" % word
        
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    trainMatrix:文档矩阵
    trainCategory: 由每篇文档类别标签所构成的向量。
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs) #计算侮辱性文档的概率，因为trainCategory取值0或1，1表示侮辱性文档。sum即为侮辱性文档个数
    #而侮辱性文档的概率为1-pAbusive ,对于多分类就不能直接用1减了。
    
    pAbusiveNum = np.ones(numWords) #侮辱性文档
    pNormalNum = np.ones(numWords) #非侮辱性文档
    
    pAbusiveDenom = 2.0
    pNormalDenom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #侮辱性文档
            pAbusiveNum += trainMatrix[i] #统计所有侮辱性文档中，所有出现词的次数
            pAbusiveDenom += np.sum(trainMatrix[i])
        else:
            pNormalNum += trainMatrix[i] #统计所有正常文档中，所有出现词的次数
            pNormalDenom += np.sum(trainMatrix[i])            
    
    pAbusiveVect = pAbusiveNum / pAbusiveDenom   # change to log() 计算的都是条件概率p(w|c)
    pNormalVect = pNormalNum / pNormalDenom     # change to log()
    
    return pAbusiveVect, pNormalVect, pAbusive
    
    
    
    
        
    
if __name__ == '__main__':
    
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print myVocabList
    
    print setOfWords2Vec(myVocabList, postingList[0])
    print setOfWords2Vec(myVocabList, postingList[1])
    print setOfWords2Vec(myVocabList, postingList[3])
    
    trainMat = []
    for postingDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
        
    pAbusiveVect, pNormalVect, pAbusive = trainNB0(trainMat, classVec)
    print pAbusiveVect
    print pNormalVect
    print pAbusive