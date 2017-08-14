#!/usr/bin/env python
# encoding=utf8

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # 节点元素名称，在构造时初始化为给定值
        self.count = numOccur # 出现次数，在构造时初始化为给定值
        self.nodeLink = None  # 指向下一个相似节点的指针，默认为None
        self.parent = parentNode  # 指向父节点的指针，在构造时初始化为给定值
        self.children = {}    # 指向子节点的字典，以子节点的元素名称为键，指向子节点的指针为值，初始化为空字典
        
    def inc(self, numOccur):
        '''
        增加节点的出现次数值
        '''
        self.count += numOccur
        
    def disp(self, ind=1):
        '''
        输出节点和子节点的FP树结构
        '''
        print '  '*ind , self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)
            
            
def createTree(dataSet, minSup=1):
    '''
    创建FP树
    dataSet：dataSet的格式比较奇特，不是直觉上得集合的list，而是一个集合的字典，以这个集合为键，值部分记录的是这个集合出现的次数。
    '''
    # 第一次遍历数据集，创建头指针表
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans] # dataSet[trans] 为此数据集出现的次数，默认都为1
            
    # 移除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    
    freqItemSet = set(headerTable.keys())
    # 空元素集，返回空
    if len(freqItemSet) == 0:
        return None, None

    # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k] , None]
    
    retTree = treeNode('Null Set', 1, None) # 根节点
    
    # 第二次遍历数据集，创建FP树
    for tranSet , count in dataSet.items():
        localD = {} # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] # 注意这个[0]，因为之前加过一个数据项
                
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]  # 根据全局频率排序，频繁的在前面
            
            updataTree(orderedItems, retTree, headerTable, count) # 更新FP树
            
    return retTree, headerTable
            
    
def updataTree(items, inTree, headerTable, count):
    '''
    修改数结构与头指针表
    items：等待插入的数据项（已删除非频繁子项，且已按全局频率排序）
    inTree: 待插入到哪个树结点下
    headerTable: 头指针表 key---->[count, treeNode link]
    count: 数据项出现次数
    '''
    if items[0] in inTree.children:
        # 有该元素项时计数值+1
        inTree.children[items[0]].inc(count)
    else:
        # 没有这个元素项时创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表或前一个相似元素项节点的指针指向新节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else: 
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
            
    if len(items) >1 :
        # 对剩下的元素项迭代调用updateTree函数
        updataTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):
    '''
    获取头指针表中该元素项对应的单链表的尾节点，然后将其指向新节点targetNode。
    '''
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
        
    nodeToTest.nodeLink = targetNode
        
        
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def findPrefixPath(basePat, treeNode):
    '''
    创建前缀路径，给定元素项生成一个条件模式基（前缀路径），这通过访问树中所有包含给定元素项的节点来完成。
    
    参数basePet：输入的频繁项
    treeNode: 当前FP树种对应的第一个节点（可在函数外部通过headerTable[basePat][1]获取）
    
    函数返回值即为条件模式基condPats，用一个字典表示，键为前缀路径，值为计数值。
    '''
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if  len(prefixPath) >1 :
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def ascendTree(leafNode, prefixPath):
    '''
    函数直接修改prefixPath的值，将当前节点leafNode添加到prefixPath的末尾，然后递归添加其父节点。
    
    最终结果，prefixPath就是一条从treeNode（包括treeNode）到根节点（不包括根节点）的路径。在主函数findPrefixPath()中再取prefixPath[1:]，即为treeNode的前缀路径。
    '''
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
        
   
def mineTree(inTree, headerTable , minSup, preFix, freqItemList):
    '''
    参数：
    inTree和headerTable是由createTree()函数生成的数据集的FP树
    minSup表示最小支持度
    preFix请传入一个空集合（set([])），将在函数中用于保存当前前缀
    freqItemList请传入一个空列表（[]），将用来储存生成的频繁项集
    
    '''
    bigL =  [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  #  对headerTable中的每个元素basePat（按计数值由小到大排序）
    
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)  #记basePat + preFix为当前频繁项集newFreqSet
        
        freqItemList.append(newFreqSet) 
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        
        myCondTree, myHead = createTree(condPattBases, minSup)
        
        if myHead != None:
            print 'conditional tree for: ', newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

            

if __name__ == '__main__':
    #rootNode = treeNode('pyramid', 9, None)
    #rootNode.children['eye'] = treeNode('eye', 13, None)
    #rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    #rootNode.disp()
    
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTable = createTree(initSet, 3)
    myFPtree.disp()
    
    #print findPrefixPath('x', myHeaderTable['x'][1])
    #print findPrefixPath('z', myHeaderTable['z'][1])
    #print findPrefixPath('r', myHeaderTable['r'][1])
    
    
    freqItems = []
    mineTree(myFPtree, myHeaderTable, 3, set([]), freqItems)
    
    print freqItems
    