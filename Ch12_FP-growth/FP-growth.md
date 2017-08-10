第12章 使用FP-growth算法来高效发现频繁项集
==========================================
FP-growth算法基于Apriori构建，但采用了高级的数据结构减少扫描次数，大大加快了算法速度。**FP-growth算法只需要对数据库进行两次扫描**，而Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，因此FP-growth算法的速度要比Apriori算法快,**通常性能要好两个数量级以上。这种算法虽然能更高效地发现频繁项集，但不能用于发现关联规则**。

FP-growth算法发现频繁项集的基本过程如下：

- 构建FP树
- 从FP树中挖掘频繁项集

**FP-growth算法**:

- **优点**：一般要快于Apriori。
- **缺点**：实现比较困难，在某些数据集上性能会下降。
- **适用数据类型**：离散型数据。

# 1 FP树：用于编码数据集的有效方式

FP-growth算法将数据存储在一种称为**FP树的紧凑数据结构**中。FP代表**频繁模式（Frequent Pattern）**。一棵FP树看上去与计算机科学中的其他树结构类似，但是它通过链接（link）来连接相似元素，被连起来的元素项可以看成一个链表。图1给出了FP树的一个例子。

![FP树](FP树.jpg)

图1 一棵FP树，和一般的树结构类似，包含着连接相似节点（值相同的节点）的连接

与搜索树不同的是，**一个元素项可以在一棵FP树种出现多次**。**FP树会存储项集的出现频率，而每个项集会以路径的方式存储在数中**。存在相似元素的集合会共享树的一部分。只有当集合之间完全不同时，树才会分叉。 树节点上给出集合中的单个元素及其在序列中的出现次数，路径会给出该序列的出现次数。

相似项之间的链接称为**节点链接（node link)**，用于快速发现相似项的位置。

举例说明，下表用来产生图1的FP树：

用于生成图1中FP树的事务数据样例

事务ID | 事务中的元素项
------ | -------------
001    | r, z, h, j, p
002    | z, y, x, w, v, u, t, s
003    | z
004    | r, x, n, o, s
005    | y, r, x, z, q, t, p
006    | y, z, x, e, q, s, t, m

**对FP树的解读**：

图1中，元素项z出现了5次，集合{r, z}出现了1次。于是可以得出结论：z一定是单独出现一次和其他符号一起出现了4次。集合{t, s, y, x, z}出现了2次，集合{t, r, y, x, z}出现了1次，z本身单独出现1次。就像这样，**FP树的解读方式是读取某个节点开始到根节点的路径。路径上的元素构成一个频繁项集，开始节点的值表示这个项集的支持度**。根据图1，我们可以快速读出项集{z}的支持度为5、项集{t, s, y, x, z}的支持度为2、项集{r, y, x, z}的支持度为1、项集{r, s, x}的支持度为1。FP树中会多次出现相同的元素项，也是因为同一个元素项会存在于多条路径，构成多个频繁项集。但是频繁项集的共享路径是会合并的，如图中的{t, s, y, x, z}和{t, r, y, x, z}

和之前一样，我们取一个最小阈值，出现次数低于最小阈值的元素项将被直接忽略。图1中将最小支持度设为3，所以q和p没有在FP中出现。

FP-growth算法的工作流程如下。首先构建FP树，然后利用它来挖掘频繁项集。为构建FP树，需要对原始数据集扫描两遍。第一遍对所有元素项的出现次数进行计数。数据库的第一遍扫描用来统计出现的频率，而**第二遍扫描中只考虑那些频繁元素**。

# 2 构建FP树

# 2.1 创建FP树的数据结构

由于树节点的结构比较复杂，我们使用一个类表示。创建文件fpGrowth.py并加入下列代码：

```python
ass treeNode:
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

>>> rootNode = treeNode('pyramid', 9, None)
>>> rootNode.children['eye'] = treeNode('eye', 13, None)
>>> rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
>>> rootNode.disp()
   pyramid   9
     eye   13
     phoenix   3

```

现在FP树所需要的数据结构已经建好，下面就可以构造FP树了。

# 2.2 构建FP树

**头指针表**

FP-growth算法还需要一个称为**头指针表的数据结构**，其实很简单，就是用来记录各个元素项的总出现次数的数组，再附带一个指针指向FP树中该元素项的第一个节点。这样每个元素项都构成一条单链表。图示说明：

![带头指针表的FP树](带头指针表的FP树.jpg)

图2 带头指针表的FP树，头指针表作为一个起始指针来发现相似元素项

这里使用Python字典作为数据结构，来保存头指针表。以元素项名称为键，保存出现的总次数和一个指向第一个相似元素项的指针。

第一次遍历数据集会获得每个元素项的出现频率，去掉不满足最小支持度的元素项，生成这个头指针表。

**元素项排序**

上文提到过，**FP树会合并相同的频繁项集（或相同的部分）**。因此为判断两个项集的相似程度需要对项集中的元素进行排序（不过原因也不仅如此，还有其它好处）。**排序基于元素项的绝对出现频率（总的出现次数）来进行**。在第二次遍历数据集时，会读入每个项集（读取），去掉不满足最小支持度的元素项（过滤），然后对元素进行排序（重排序）。

对示例数据集进行过滤和重排序的结果如下：

事务ID | 事务中的元素项 | 过滤(删除不满足最小支持度的元素hjpwvuonqem)及重排序后的事务
------ | -------------  | --------------------
001    | r, z, h, j, p	| z, r
002    | z, y, x, w, v, u, t, s	| z, x, y, s, t
003    | z              | z
004    | r, x, n, o, s  | x, s, r
005    | y, r, x, z, q, t, p | z, x, y, r, t
006    | y, z, x, e, q, s, t, m | z, x, y, s, t

**构建FP树**

在对事务记录过滤和排序之后，就可以构建FP树了。从空集开始，将过滤和重排序后的频繁项集一次添加到树中。如果树中已存在现有元素，则增加现有元素的值；如果现有元素不存在，则向树添加一个分支。对前两条事务进行添加的过程：

![FP树构建过程示意](FP树构建过程示意.jpg)

图3 FP树构建过程示意（添加前两条事务）

算法：构建FP树

```code
输入：数据集、最小值尺度
输出：FP树、头指针表
1. 遍历数据集，统计各元素项出现次数，创建头指针表
2. 移除头指针表中不满足最小值尺度的元素项
3. 第二次遍历数据集，创建FP树。对每个数据集中的项集：
    3.1 初始化空FP树
    3.2 对每个项集进行过滤和重排序
    3.3 使用这个项集更新FP树，从FP树的根节点开始：
        3.3.1 如果当前项集的第一个元素项存在于FP树当前节点的子节点中，则更新这个子节点的计数值
        3.3.2 否则，创建新的子节点，更新头指针表
        3.3.3 对当前项集的其余元素项和当前元素项的对应子节点递归3.3的过程
```

代码（在fpGrowth.py中加入下面的代码）：

1 总函数：createTree

```python

```

