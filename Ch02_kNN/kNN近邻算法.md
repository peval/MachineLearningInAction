第2章 K-近邻算法（KNN）
==================
# 工作原理与流程：
KNN算法采用 **测量不同特征值之间的距离** 进行分类。

- 准备训练样本集，要求每条记录都打好标签（提前分好类）；
- 输入无标签的数据，计算输入与样本集中每条记录的距离；
- 距离排序，选取k个最近邻的数据记录与其标签；（k通常选不大于20的整数）
- k个选取标签中出现次数最多的即为输入数据的分类标签。

**优点**：精度高、对异常值不敏感、无数据输入假定。

**缺点**：计算复杂度高、空间复杂度高。

**适用数据范围**：数值型与标称型。

___
# 案例说明
## 案例一、约会预测
海伦共收集约会网站上1000条数据，其中每个数据包含3种特征
- 每年获得的飞行里程数
- 玩视频游戏所花费时间百分比
- 每周消费的冰淇淋公升数

海伦将这1000条数据分别进行标记，是否为自己喜欢的类型。所有特征与标签数据都存放在datingTestSet.txt文件中。

交往过的3种标记类型：
- 不喜欢的人
- 魅力一般的人
- 极具魅力的人

### 从文本中读取并解析数据
```python
def file2matrix(filename):
    '''
    从文件filename中读取约会数据样本，并返回样本数据与标签
    '''
    returnMat = []
    returnLabel = []
    lable = {'didntLike':0, 'smallDoses':1, 'largeDoses':2}
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
```
代码中并没有显示的将文件中的字符串转换成二组数组中的int或float，而是通过zeros初始化一个数组，并指定数值类型。由numpy在内部进行类型转化。

### 从文本中读取并解析数据
```python
    datingDataMat , datingLabels = file2matrix("datingTestSet.txt")
    print datingDataMat
    print datingLabels[0:20]
    
输出
    [[  4.09200000e+04   8.32697600e+00   9.53952000e-01]
    [  1.44880000e+04   7.15346900e+00   1.67390400e+00]
    [  2.60520000e+04   1.44187100e+00   8.05124000e-01]
    ..., 
    [  2.65750000e+04   1.06501020e+01   8.66627000e-01]
    [  4.81110000e+04   9.13452800e+00   7.28045000e-01]
    [  4.37570000e+04   7.88260100e+00   1.33244600e+00]]
    [2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2]

```

### 参考
[Numpy使用中文教程](http://old.sebug.net/paper/books/scipydoc/numpy_intro.html)

[Numpy使用英文教程](https://docs.scipy.org/doc/numpy/reference/)




