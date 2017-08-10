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
            
            
if __name__ == '__main__':
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()
    