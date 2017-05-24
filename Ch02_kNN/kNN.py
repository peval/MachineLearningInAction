#!/usr/bin/evn python
# encoding=utf8

'''
k-近邻（kNN）算法测试
'''


import numpy as np   #科学计算模块包
import operator #运算符模块包

def createDataset():
    '''
    准备训练样本集，返回数据与对应标签
    '''
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
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
    dataSetLen = dataSet.shape[0] #dataSet.shape返回dataSet样本的维数元组，其中第一个元素为样本个数
    
    diffMat = np.tile(inX, (dataSetLen, 1)) - dataSet
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
    '''
    tile也可以使用repeat代替
    >>> bb =np.array(inX)
    >>> bb.shape
    (2,)
    >>> bb.shape = (1,-1)
    >>> bb
    array([[0, 0]])
    >>> bb.repeat(dataSetLen,axis=0)
    array([[0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]])
    '''
    
    
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    #sort distances
    sortedDistIndicies = distances.argsort() #返回从小到大排序后的index
    
    classCount = {}
    for i in range(k):
        voteLable = lables[sortedDistIndicies[i]]
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
        
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
    

def file2matrix(filename):
    '''
    从文件filename中读取约会数据样本，并返回样本数据与标签
    '''
    returnMat = []
    returnLabel = []
    lable = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        returnMat = np.zeros((len(lines), 3)) #定义一个(len(lines), 3) 2维数组并填充0
        '''
        这里使用zeros初始化全是0是二维数组的好处是，np会默认将从文件读取的字符串转化成float, 不用在代码里显示转换
        '''
        index = 0
        for line in lines:
            listFromLine = line.strip().split('\t')
            returnMat[index,:] = listFromLine[0:3]
            returnLabel.append(lable[listFromLine[-1]] if listFromLine[-1] in lable else 1)
            index +=1
        returnMat = np.array(returnMat)
            
    return returnMat , returnLabel
    
def autoNorm(dataSet):
    '''
    针对样本数据归一化，减小不同属性数据单位不一致，导致权重问题
    '''
    mimVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - mimVal
    normDataSet = np.zeros(dataSet.shape)
    
    normDataSet = dataSet - np.tile(mimVal, (dataSet.shape[0], 1))
    normDataSet = normDataSet / np.tile(ranges, (dataSet.shape[0], 1))
    
    
    return normDataSet, ranges, mimVal, maxVal


def datingClassTest():
    '''
    测试约会KNN算法在此样本下的正确率。使用90%的总样本作为训练样本，其余10%用于评估算法的正确率
    '''
    hoRatio = 0.05
    datingDataMat , datingLabels = file2matrix("datingTestSet.txt")
    datingDataMat, ranges, mimVal, maxVal = autoNorm(datingDataMat)
    numTestVecs = int(hoRatio * datingDataMat.shape[0])
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifyResultLabel = classify0(datingDataMat[i,:], datingDataMat[numTestVecs:datingDataMat.shape[0],:], datingLabels[numTestVecs:datingDataMat.shape[0]], 10)
        print 'the classify came back with: %d, the real answer is : %d' % (classifyResultLabel, datingLabels[i])
        if (classifyResultLabel != datingLabels[i]) : errorCount += 1.0
    print 'the total error rate is : %f' % (errorCount/float(numTestVecs))
    
    
    
def classifyPersion():
    '''
    命令行输入待预测的数据，输出预测结果。
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    
    datingDataMat , datingLabels = file2matrix("datingTestSet.txt")
    datingDataMat, ranges, mimVal, maxVal = autoNorm(datingDataMat)
    
    inArr = np.array([percentTats, ffMiles, iceCream])
    inArr = (inArr - mimVal)/ranges
    
    classifyResultLabel = classify0(inArr, datingDataMat, datingLabels, 10)
    print 'You will probably like this person: ', resultList[classifyResultLabel]
    
    
    

def img2vector(filename):
    '''
    图像转换成向量的函数img2vector：该函数创建1*1024的Numpy数组，然后打开给定文件，循环读取文件的前32行>，并将每行的头32个字符存储在Numpy数组中，最后返回数组。
    '''
    returnVect = np.zeros((1,1024))
    with open(filename, 'r') as fp:
        index = 0
        for line in fp.readlines():
            line = line.strip()
            for i in range(32):
                returnVect[0,index*32+i] = int(line[i])
            index +=1
    return returnVect
        
    
    
    
if __name__ == "__main__":
    
    dataSet, lables = createDataset()
    inX = [0, 0]
    #print classify0(inX, dataSet, lables, 3)
    
    #datingDataMat , datingLabels = file2matrix("datingTestSet.txt")
    #print datingDataMat
    #print datingLabels[0:20]
    
    #datingClassTest()
    
    #classifyPersion()
    
    
    

    #import matplotlib.pyplot as plt
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    #plt.show()
    
    print img2vector('trainingDigits/8_60.txt')

