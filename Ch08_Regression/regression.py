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

def plotStandRegress():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xArr , yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    
    ws = standRegres(xArr, yArr)
    yHat = xMat*ws #回归函数，用于预测
    
    #np.corrcoef(yHat.T, yMat)
    
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) #绘出数据集散点图
     
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()
    

def lwlr(testPoint , xArr , yArr ,k=1.0):
    '''
        局部加权线性回归函数
        testPoint: 待预测的点
        xArr: 样本特征
        yArr: 样本标签值
        k默认为1.0
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0] #样本数
    weights = np.mat(np.eye((m)))  #创建对角矩阵,生成m*m单位矩阵
    
    for j in range(m):
        diffMat = testPoint - xMat[j,:]   #  diffMat * diffMat.T  就是两个样本点之间距离的平方
        weights[j,j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2)) #权重值大小以指数级衰减
    
    xTx = xMat.T * (weights * xMat)
    
    if np.linalg.det(xTx) == 0.0: #计算行列式，判断是否有逆矩阵。若没检查行列式是否为0就直接计算矩阵的逆，将会出现错误。另外Numpy的线性代数库还提供一个函数来解未知矩阵，修改ws = xTx.I * (xMat.T * yMat)为ws = np.linalg.solve(xTx, xMat.T*yMat)
        print "This matrix is singular , cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    
    return testPoint * ws


def lwlrTest(testArr , xArr, yArr, k=1.0):
    '''
    '''
    m = np.shape(testArr)[0] #待预测测试样本个数
    yHat = np.zeros(m)
    
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    
    return yHat

def plotLwlrRegress():
    '''
    '''
    import matplotlib.pyplot as plt
    plt.figure(u"局部加权线性回归函数,使用不同的k值[1.0, 0.01, 0.003]")
    
    plt.subplot(311)
    
    xArr , yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:, 0,:]
    
    yHat = lwlrTest(xArr , xArr, yArr, k=1.0)
    plt.plot(xSort[:,1], yHat[srtInd])

    plt.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.text(0.5, 4.0, 'k=1.0', color='red', size=16, horizontalalignment='right', verticalalignment='bottom')
    
    plt.subplot(312)
    yHat1 = lwlrTest(xArr , xArr, yArr, k=0.01)
    plt.plot(xSort[:,1], yHat1[srtInd])
    plt.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='blue')
    plt.text(0.5, 4.0, 'k=0.01', color='red', size=16, horizontalalignment='right', verticalalignment='bottom')
    
    
    plt.subplot(313)
    yHat2 = lwlrTest(xArr , xArr, yArr, k=0.003)
    plt.plot(xSort[:,1], yHat2[srtInd])
    plt.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='green')   
    plt.text(0.5, 4.0, 'k=0.003', color='red', size=16, horizontalalignment='right', verticalalignment='bottom')
    
    
    plt.show()
    
        



if __name__ == '__main__':
    xArr , yArr = loadDataSet('ex0.txt')
    print xArr
    print yArr
    
    #ws = standRegres(xArr, yArr)
    #print ws
    
    #plotStandRegress()
    
    
    # 局部加权线性回归函数
    print yArr[0]
    print lwlr(xArr[0], xArr, yArr, 1.0)
    print lwlr(xArr[0], xArr, yArr, 0.001)
    
    plotLwlrRegress()
    