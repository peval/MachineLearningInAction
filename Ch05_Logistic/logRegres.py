#!/usr/bin/env python
# encoding=utf8

import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    with open('testSet.txt', 'r') as fp:
        for line in fp.readlines():
            lineAttr = line.strip().split()
            dataMat.append([1.0, float(lineAttr[0]), float(lineAttr[1])])
            labelMat.append(int(lineAttr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    '''
    sigmoid函数（阶跃函数），当x=0时，sigmoid函数值为0.5，随着x的增大，对应sigmoid函数值将逼近1；随着x的减小，对应sigmoid函数值将逼近0；
    '''
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    
    alpha = 0.001 #向目标移动的步长
    maxCycles = 500 #迭代次数
    weights = np.ones((n, 1))
    
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    '''
    '''
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    
    n = np.shape(dataArr)[0]
    
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    
    x = np.arange(-3.0, 3.0, 0.1)
    y = (- weights[0] - weights[1] * x) / weights[2]
    
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
    
def lineLogRegress():
    '''
    线性回归测试,
    '''
    import matplotlib.pyplot as plt
    time = [ 1, 2, 3, 4, 5, 6, 7, 8, 9] #原始数据为物理中的自由落体打点实验，求重力加速度g
    speed = [0.199, 0.389, 0.580, 0.783, 0.980, 1.177, 1.380, 1.575, 1.771] 
    
    A = np.vstack([time ,np.ones(len(time))]).T
    
    a, b = np.linalg.lstsq(A, speed)[0]
    
    
    timeArr = np.array(time)
    speedArr = np.array(speed)
    
    plt.plot(timeArr, speedArr, 'o', label='Original data', markersize=10)
    plt.plot(timeArr, a * timeArr + b , 'r', label='Fitted line')
    
    plt.show()
    


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    #plotBestFit(weights.getA())
    lineLogRegress()