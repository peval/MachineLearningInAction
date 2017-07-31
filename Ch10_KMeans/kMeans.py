#!/usr/bin/env python
# encoding=utf8

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    '''
    从文件中加载数据
    '''
    dataMat = []
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine)
            dataMat.append(fltLine)
    return dataMat

def distEclud(vecA , vecB):
    '''
    距离计算公式，相异度计算，这里选用欧几里得距离
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB , 2))) # 欧几里得距离

def randCent(dataSet, k):
    '''
    从dataSet中随机选择k个质心，质心必须要在整个数据集的边界之内。
    这里通过找到数据集中每一维的最小和最大值来完成。然后生成0~1之间的随机数 乘以 维范围，以确保随机点在数据的边界之内。
    '''
    n = np.shape(dataSet)[1] #列数、特征个数
    centroids = np.mat(np.zeros((k,n)))
    
    for j in range(n):
        minJ = np.min(dataSet[:,j])  #特征J列最小值
        rangeJ = np.max(dataSet[:,j]) - minJ
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)   # np.random.rand(k, 1) 生成k个0~1之间的随机数
    return centroids
        
    
def kMeans(dataSet, k, distMeas = distEclud, createCent=randCent):
    '''
    K-均值聚类算法。k为簇的数目，distMeas距离公式，createCent质心选择算法
    '''
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2))) #簇分配结果矩阵，包含两列，第一列记录簇索引值，第二列存储误差（指当前点到簇质心的距离，
    #后面会使用这个误差来评价聚类的效果）。
    
    centroids = createCent(dataSet, k)
    clusterChanged = True #标志变量，是否继续迭代
    
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            
            for j in range(k):
                distJI = distEclud(dataSet[i,:], centroids[j,:]) # 计算点i到质心j的距离
                if distJI < minDist: #寻找最近的质心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex, minDist**2  #簇分配结果矩阵,存储簇索引值与误差
        print centroids
        
        #待一轮原始数据都找到对应质心后，然后遍历所有质心并重新计算簇质心位置，最后继续下一轮迭代
        for cent in range(k):
            #先通过数组过滤来获得给定簇的所有点，然后计算所有点的均值，选项axis =0表示沿矩阵的列方向进行均值计算；最后，程序返回所有的类质心与点分配结果。
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]   # .A 将matrix转换成array; np.nonzero()返回的是非0值的索引数组
            #dataSet[[2,3,4]] 返回第2，3，4行
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #沿矩阵的列方向计算平均值，最终输出一行多列。若axis=1，最终输出多行一列
    return centroids, clusterAssment
    
 
def platKMeans(dataMat, myCentroids, clusterAssing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    scatterMarkers=['D', 'o', '^', '*', 'p']  #Markers 点样式取值 https://matplotlib.org/api/markers_api.html
    for i in range(len(myCentroids)):
        ptsInClust = dataMat[np.nonzero(clusterAssing[:,0] == i)[0]]
        plt.scatter(ptsInClust[:,0], ptsInClust[:,1],  marker=scatterMarkers[i],s=10 , linewidths=3)
        
    plt.scatter(myCentroids[:,0], myCentroids[:,1], s=169,  marker='x', linewidths=3)
    plt.show()
    
def biKmeans(dataSet, k ,distMeas = distEclud):
    '''
    二分K-均值算法
    '''
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] # 计算原数据的质心，也就是所有数据的均值
    centList = [centroid0] #用于存储所有质心点
    
    for j in range(m):
        clusterAssment[j,1] = distMeas(dataSet[j, :], np.mat(centroid0)) ** 2
    
    while (len(centList) < k) :
        lowestSSE  = np.inf  #无穷大infinite, 无穷小-inf
        
        for i in range(len(centList)): #对每个簇划分成两个子簇，并计算其SSE误差。比较所有划分，选出一个SSE减少最多的簇划分，作为本次迭代的最优划分。
            ptsInCurrClusser = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0], :] #当前簇的所有数据
            centroidMat, splitClustAss = kMeans(ptsInCurrClusser, 2, distMeas) #划分成两个子簇
            
            sseSplit = np.sum(splitClustAss[:,1]) #计算两个子簇的SSE误差
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A !=i )[0] , 1]) #计算非此划分簇以外的数据的SSE误差
            
            if (sseSplit + sseNotSplit) <  lowestSSE:  #选择误差减小最多的簇划分
                bestCentToSplit = i
                bestNewCents = centroidMat.tolist()
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                
        # bestClustAss作为本次迭代的最优划分，由于是k=2的子簇划分，其索引值取值0或1.
        # 以下两条代码是，更新索引值为总centList质心点数组是的序列号。
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList) #将子簇中索引为1的，修改成新的最大序列（质心点数组大小+1）
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #将子簇中索引为0的，修改成原簇对应的序列
        
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        
        centList[bestCentToSplit] = bestNewCents[0]  #对应上面bestClustAss的修改，0子簇使用原centList中的序列号
        centList.append(bestNewCents[1])  #对应上面bestClustAss的修改，1子簇在centList数组中新增加一个序列号
        
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss  #修改新误差到clusterAssment中
        
    return np.mat(centList), clusterAssment
                
                
            
            
            
            

if __name__ == '__main__':
    dataMat = np.mat(loadDataSet('testSet.txt'))
    randCent(dataMat, 2)
    
    #myCentroids, clusterAssing = kMeans(dataMat, 4)
    #platKMeans(dataMat, myCentroids, clusterAssing)
    
    dataMat2 = np.mat(loadDataSet('testSet2.txt'))
    
    myCentroids, clusterAssing = biKmeans(dataMat2,3)
    print myCentroids
    platKMeans(dataMat2, myCentroids, clusterAssing)
    



