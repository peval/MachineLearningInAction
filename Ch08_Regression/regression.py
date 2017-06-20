#!/usr/bin/env python
# encoding=utf8

import numpy as np

def loadDataSet(filename):
    dataMat = []; labelMat = []
    with open(filename, 'r') as fp: 
        for line in fp.readlines():
            lineAttr = line.strip().split()
            
            dataMat.append([float(attr) for attr in lineAttr[:-1]]) # 由于python数组index是以0开始，样本文件第一列都为1.0，即x0等于1.0，第二列的值为x1
            labelMat.append(float(lineAttr[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    使用普通最小二乘法，来计算最佳拟合直线
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    
    if np.linalg.det(xTx) == 0.0: #计算行列式，判断是否有逆矩阵。若没检查行列式是否为0就直接计算矩阵的逆，将会出现错误。另外Numpy的线性代数库还提供一个函数来解未知矩阵，修改ws = xTx.I * (xMat.T * yMat)为ws = np.linalg.solve(xTx, xMat.T*yMat)
        print "This matrix is singular , cannot do inverse"
        return
    
    ws = xTx.I * (xMat.T * yMat) # ws存放回归系数
    return ws

def plotRegress():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xArr , yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    
    ws = standRegres(xArr, yArr)
    yHat = xMat*ws #回归函数，用于预测
    
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) #绘出数据集散点图
     
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()



if __name__ == '__main__':
    xArr , yArr = loadDataSet('ex0.txt')
    print xArr
    print yArr
    
    ws = standRegres(xArr, yArr)
    print ws
    
    plotRegress()
    