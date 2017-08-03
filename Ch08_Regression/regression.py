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
        testPoint: 待预测的点，单条记录
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
    局部加权线性回归函数
        testArr: 待预测的点，多条记录
        xArr: 样本特征
        yArr: 样本标签值
        k默认为1.0
    '''
    m = np.shape(testArr)[0] #待预测测试样本个数
    yHat = np.zeros(m)
    
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    
    return yHat

def plotLwlrRegress():
    '''
    画出局部加权线性回归函数生成的回归直线
    '''
    import matplotlib.pyplot as plt
    plt.figure(u"局部加权线性回归函数,使用不同的k值[1.0, 0.01, 0.003]") #图的title
    
    plt.subplot(311) #311表明，整个图分为三行一列，此时在第一行内画图
    
    xArr , yArr = loadDataSet('ex0.txt')
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:, 0,:]
    
    yHat = lwlrTest(xArr , xArr, yArr, k=1.0)
    plt.plot(xSort[:,1], yHat[srtInd])

    plt.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.text(0.5, 4.0, 'k=1.0', color='red', size=16, horizontalalignment='right', verticalalignment='bottom') #在线条边标注k值
    
    plt.subplot(312) #312表明，整个图分为三行一列，此时在第二行内画图
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
    
    
# 预测鲍鱼的年龄

def rssError(yArr , yHatArr):
    '''
    计算预测值与真实值之间的误差，平方和
    '''
    return ((yArr - yHatArr)**2).sum()
        
def abaloneTest():
    '''
    使用LWLR预测鲍鱼的年龄
    '''
    xArr , yArr = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
    yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
    yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
    
    print rssError(yArr[0:99], yHat01.T) #56.8041972556
    print rssError(yArr[0:99], yHat1.T)  #429.89056187
    print rssError(yArr[0:99], yHat10.T) #549.
    
    
    yHat01 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.1)
    yHat1 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
    yHat10 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)
    
    print rssError(yArr[100:199], yHat01.T) #106085.423168
    print rssError(yArr[100:199], yHat1.T)  #573.52614419
    print rssError(yArr[100:199], yHat10.T) #517.571190538   
    
    
    ws = standRegres(xArr[0:99], yArr[0:99])
    yHat = np.mat(xArr[100:199]) * ws
    print rssError(yArr[100:199], yHat.T.A) #518.636315325   
    
    
#岭回归
    
def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    岭回归缩减方法,用于计算回归系数
    xMat:样本特征
    yMat: 样本标签值
    lam: lambda值，默认为0.2
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam #  np.eye(np.shape(xMat)[1])生成一个特征个数n*特征个数n的单位矩阵
    
    if np.linalg.det(denom) == 0.0: #计算行列式，判断是否有逆矩阵。若没检查行列式是否为0就直接计算矩阵的逆，将会出现错误。另外Numpy的线性代数库还提供一个函数来解未知矩阵，修改ws = xTx.I * (xMat.T * yMat)为ws = np.linalg.solve(xTx, xMat.T*yMat)
        print "This matrix is singular , cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr , yArr):
    '''
    测试岭回归缩减方法，通过使用不同的lambda值
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    
    #下面进行特征数据的归一化标准处理，使每个维具有相同的重要性
    yMean = np.mean(yMat,0) #计算yMat，类型的平均值
    yMat = yMat - yMean
    
    #所有特征都减去各自的均值，并除以方差。
    xMeans = np.mean(xMat , 0) #计算每个特征的平均值
    xVar = np.var(xMat, 0) #计算指定轴上的方差。0表示二维数组中计算a11、a21、a31、……、am1的方差， 依此类推
    xMat = (xMat - xMeans) / xVar
    
    numTestPts = 30 #使用30个lambda值进行测试，且lambda值是以指数变化。这样可以看出lambda在取非常小与非常大的值时分别对结果造成的影响。
    wMat = np.zeros((numTestPts, np.shape(xMat)[1])) #生成一个30*特征个数的0矩阵
    
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, lam= np.exp(i - 10))
        wMat[i] = ws.T
    return wMat

def plotRidge():
    '''
    画出30组指数变化的lambda值生成的岭回归直线
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure(u"30组指数变化的lambda值生成的岭回归直线") #图的title
    
    ax = fig.add_subplot(111) #311表明，整个图分为三行一列，此时在第一行内画图
    
    xArr , yArr = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(xArr, yArr)    
    
    ax.plot(ridgeWeights)
    plt.show()
    
    
    
    
    



if __name__ == '__main__':
    xArr , yArr = loadDataSet('ex0.txt')
    print xArr
    print yArr
    
    #ws = standRegres(xArr, yArr)
    #print ws
    
    #plotStandRegress()
    
    
    # 局部加权线性回归函数
    #print yArr[0]
    #print lwlr(xArr[0], xArr, yArr, 1.0)
    #print lwlr(xArr[0], xArr, yArr, 0.001)
    
    #plotLwlrRegress()
    
    
    # 预测鲍鱼的年龄
    #abaloneTest()
    
    
    plotRidge()
    