#! /usr/bin/env python
# encoding:utf8

import matplotlib.pyplot as plt

#定义树节点常量 文本框、箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    绘制带箭头的注解
    '''
    #createPlot.ax1 全局变量，指向一个绘图区
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction", \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, \
                            arrowprops=arrow_args)
    
def plotMidText(cntrPt, parentPt, txtString):
    '''
    在父子节点之间填充文本信息
    '''
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    
    createPlot.ax1.text(xMid, yMid, txtString)
    

def plotTree(myTree, parentPt, nodeTxt):
    '''
    画决策树
    '''
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff  - 1.0/plotTree.totalD
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
            
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    
    #创建一个新的图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点。
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    
    axprops = dict(xticks=[], yticks=[])
    
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    
    plotTree.xOff = -0.5/plotTree.totalW ;
    plotTree.yOff = 1.0
    
    plotTree(inTree, (0.5, 1.0), U'决策树')
    plt.show()
    
    
def getNumLeafs(myTree):
    '''
    获取叶子节点数
    '''
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys() :
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取树的层数
    '''
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys() :
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    '''
    返回预先存储的树信息
    '''
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]
    
    

if __name__ == '__main__':
    #createPlot()
    
    myTree = retrieveTree(0)
    print myTree

    print getNumLeafs(myTree)
    print getTreeDepth(myTree)
    
    #createPlot(myTree)
    
    

