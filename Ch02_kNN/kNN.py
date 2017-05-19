#!/usr/bin/evn python
# encoding=utf8

'''
k-近邻（kNN）算法测试
'''


import numpy    #科学计算模块包
import operator #运算符模块包

def createDataset():
    '''
    准备训练样本集，返回数据与对应标签
    '''
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, lables, k):
    '''
    kNN算法核心：
    1.计算输入inX与训练样本dataSet中每个记录之间的距离；
    2.按距离递增排序；
    3.选取与当前输入inX距离最近的k个点；
    4.确定前k个点所在类别标签出现频率；
    5.返回前k个点出现频率最高的类别标签，作为输入inX记录的预测分类；
    
    参数说明：
    inX：输入要进行预测的数据记录。
    dataSet:训练样本
    lables:训练样本对应标签（与dataSet len 等长）
    k: 要选取前多少个相近记录。
    '''
    dataSetLen = dataSet.shape[0] #返回dataSet样本长度
    
    diffMat = numpy.tile(inX, (dataSetLen, 1)) - dataSet
    '''
    tile函数说明: Repeat elements of an array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])
    '''
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    #sort distances
    sortedDistIndicies = distances.argsort()
    
    classCount = {}
    for i in range(k):
        voteLable = lables[sortedDistIndicies[i]]
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
        
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
    
    
    
if __name__ == "__main__":
    
    dataSet, lables = createDataset()
    inX = [0, 0]
    print classify0(inX, dataSet, lables, 3)
    

