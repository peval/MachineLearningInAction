#!/usr/bin/env python
# encoding=utf8

from math import log
import operator
import treePlotter


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


def splitDataSet(dataSet, axis, value):
    '''
    dataSet: 待划分的数据集
    axis： 数据集中的某一特征
    value： 特征某个取值
    
    返回数据集中特征axis值为value的子数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
            
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    循环计算信息熵，选择最优的划分方式
    '''
    numFeatures = len(dataSet[0]) -1  #最后一个属性是标签
    baseEntropy = calcShannonEnt(dataSet) #计算原始数据集的信息熵，此信息熵应是最大的
    bestFeatures = -1
    bestInfoGain = 0.0
    
    
    for i in range(numFeatures): #遍历所有特征
        featureValue = [feature[i] for feature in dataSet]
        uniqFeatureValue = set(featureValue)
        
        newEntropy = 0.0
        for value in uniqFeatureValue:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            
        infoGain =  baseEntropy - newEntropy
        
        if  infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatures = i
            
    return bestFeatures

def majorityCnt(classList):
    '''
    采用多数表决的方法决定该叶子节点的分类，即classList中出现次数最多的lable
    classList: 划分后的子数据集的标签集合
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    创建决策树，dataSet为数据集
    lables: 为属性名称集合
    '''
    classList = [feature[-1] for feature in dataSet]
    if classList.count(classList[0]) == len(classList): #当数据集中的数据都是同一类型时
        return classList[0]
    if len(dataSet[0]) == 1:   #dataSet中只有标签类型时，即所有特征都已划分完。则选择出现次数最多的标签类型
        return majorityCnt(classList)
    
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLable = labels[bestFeature]
    tmpLabels = labels[:] #复制一份,避免修改原始labels
    del(tmpLabels[bestFeature])
    
    myTree = {bestFeatureLable:{}}
    featureValue = [feature[bestFeature] for feature in dataSet]
    uniqFeatureValue = set(featureValue)
    
    for value in uniqFeatureValue:
        subLables = tmpLabels[:] #重新复制一份labels，避免各分支调用createTree修改labels相互影响
        myTree[bestFeatureLable][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLables)
        
    return myTree

def classify(inputTree, featLabels , testVec):
    '''
    决策树分类过程.
    inputTree:决策树
    featLabels: 特征名称向量,用于将特征名称转换成索引
    testVec: 待分类的数据行
    '''
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if key == testVec[featIndex]:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    myDat, labels = createDataset()
    print calcShannonEnt(myDat)
    
    print splitDataSet(myDat, 0, 1)
    print splitDataSet(myDat, 0, 0)
    
    
    print chooseBestFeatureToSplit(myDat)
    
    print createTree(myDat, labels)
    
    #myDat[0][-1] = 'maybe'
    #print myDat
    #print calcShannonEnt(myDat)
    
    myTree = treePlotter.retrieveTree(0)
    print myTree
    
    print classify(myTree, labels, [1,0])
    print classify(myTree, labels, [1,1])
    
    
    

        
    