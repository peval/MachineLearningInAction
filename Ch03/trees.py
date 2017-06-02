#!/usr/bin/env python
# encoding=utf8

from math import log


'''
 决策树 算法测试
'''

def calcShannonEnt(dataset):
    '''
    计算dataset数据集的熵（香农，信息熵）
    '''
    numEntries = len(dataset)
    labelCounts = {}
    
    for featVec in dataset:
        label = featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        
        labelCounts[label] +=1
            
    shannonEnt = 0.0
    for label in labelCounts:
        prob = float(labelCounts[label]) / numEntries #使用所有类标签的发生频率计算类别出现的概率
        shannonEnt -= prob * log(prob, 2)
        
    return shannonEnt

def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               ]
    labels = ['no surfacing', 'flippers']
    
    return dataSet, labels


if __name__ == '__main__':
    dataSet, labels = createDataset()
    print calcShannonEnt(dataSet)

        
    