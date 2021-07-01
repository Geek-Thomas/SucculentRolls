#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: GeekThomas
# time: 2021/6/30
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: GeekThomas
# time: 2021/6/29
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 定义文本框和箭头格式（树节点格式的常量）
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrows_args = dict(arrowstyle='<-')

# 加载数据集
def createDataset():
    """
    :return: dataSet是二维列表，其中最后一列为分类标签 features是特征列表
    """
    df = pd.read_csv('example_data.csv')
    dataSet = np.array(df.values)
    features = df.columns.tolist()[: -1]
    return dataSet, features

def calGini(dataset):
    """
    输入数据集，获得数据集基尼系数
    :param dataset: 包括初始数据集和划分后数据集
    :return: 基尼系数（浮点数）
    """
    # 获取标签类别数目，标签是最后一列
    y_labels = np.unique(dataset[:, -1])
    y_counts = len(dataset)   # 总的数据数
    y = {}
    gini = 1.0
    for y_label in y_labels:
        # 计算数据集中每个类别出现的频率/概率
        y[y_label] = len(dataset[dataset[:, -1] == y_label]) / y_counts
        # 计算此类别基尼系数
        gini -= y[y_label] ** 2
    return gini

def splitDataset(dataset, i, value, types=1):
    """
    根据传入条件（dataset第i列的value）进行划分子集
    :param dataset: 包括初始数据集和划分后数据集
    :param i: dataset第i列
    :param value: 根据第i列的值进行划分
    :param types: 根据输入的types，标记划分后的数据集（是value，不是value）
    :return: 返回根据第i个特征划分后的子集及对应子集中数据个数
    """
    # 根据本列中值=value进行划分
    if types == 1:
        subDataset = dataset[dataset[:, i] == value]
    # 根据本列中值!=value进行划分
    if types == 2:
        subDataset = dataset[dataset[:, i] != value]
    return subDataset, len(subDataset)

def chooseBestFeature(dataset):
    """
    返回最优划分特征以及每个特征对应的基尼系数
    :param dataset:
    :return:
    """
    numTotal = len(dataset)   # 记录下dataset数据量
    numFeatures = dataset.shape[1] - 1 # 特征数量
    bestFeature = -1      # 初始化一个最优特征，实际上这个特征应该从0开始
    Gini = {}             # 使用Gini存储每一列中每个value的基尼系数
    for i in range(numFeatures):
        # 这个i对应splitDataset中的i
        # 每一列中对应多个value值
        values = dict(Counter(dataset[:, i]))
        for value in values.keys():
            featureGini = 0.0  # 因为一个特征可能存在两个以上特征值，需要记录每次二分后的gini系数
            # 对某一列x中，会出现x=是，y=是的特殊情况，这种情况下按“是”、“否”切分数据得到的Gini都一样，
            # 设置此参数将所有特征都乘以一个比1大一点点的值，但按某特征划分Gini为0时，设置为1
            bestFlag = 1.001
            subDataset1, length1 = splitDataset(dataset, i, value, 1)
            subDataset2, length2 = splitDataset(dataset, i, value, 1)
            if length1 == 0: # 划分出为同一类时
                bestFlag = 1
            # 计算此列特征划分后的基尼系数
            featureGini += length1 / numTotal * calGini(subDataset1) + length2 / numTotal * calGini(subDataset2)
            Gini[f'{i}_{value}'] = featureGini * bestFlag
    # 取基尼系数最小的特征为最优划分特征
    bestFeature = sorted(Gini.items(), key=lambda x: x[1])[0][0]
    return bestFeature, Gini


def createTree(dataset, features):
    #     print(features)
    y_labels = np.unique(dataset[:, -1])
    # 情况1：如果只有一类标签，那么直接返回该类
    if len(y_labels) == 1:
        return y_labels[0]
    # 情况2：如果特征集为空，那么就返回类别多的那一类
    if len(dataset[0]) == 1:
        return Counter(dataset[:, -1]).most_common()[0][0]
    # 情况3：根据各特征的基尼系数，划分数据集，再筛选出最优特征
    bestFeature, gini = chooseBestFeature(dataset)

    # 因为得到的最优特征是i_value，所以需要拆分
    bestFeatureCol = int(bestFeature.split('_')[0])
    #     print(bestFeatureCol)
    #     print(features)
    bestFeatureLabel = features[bestFeatureCol]
    # 以最优特征为根节点，构造节点树
    tree = {bestFeatureLabel: {}}
    # 删除特征集中的最优特征
    del features[bestFeatureCol]

    # 使用最优特征划分子树，会得到两个节点，需要统计最优划分特征所属类别，将其设为左子树
    bestFeatureValue = bestFeature.split('_')[1]
    #     print(bestFeatureLabel)
    y_label_split = dataset[dataset[:, bestFeatureCol] == bestFeatureValue][:, -1]  # 得到最优特征划分后的标签
    # 出现概率最大的即为最优特征所属分类（这里使用频数来衡量）
    y_leaf = Counter(y_label_split).most_common()[0][0]
    tree[bestFeatureLabel][bestFeatureValue] = y_leaf

    # 在划分完最有特征后，需要删除该列，然后再递归调用
    datasetNew = np.delete(dataset, bestFeatureCol, axis=1)  # 注意原dataset不受影响
    # 划分后特征集
    subFeatures = features[:]  # 对subFeatures进行操作不会影响到features
    #     print(subFeatures)
    # 前面我们设定了左子树，现在需要设定右子树，因为CART获得的是二叉树，所以二叉树的节点应该是y_labels的两个值
    y1, y2 = y_labels[0], y_labels[1]
    # 我们这里的y_labels是["no", "yes"]
    if y1 == y_leaf:
        # 那么我们需要将右子树的值设定为y2
        tree[bestFeatureLabel][y2] = {}
        tree[bestFeatureLabel][y2] = createTree(datasetNew, subFeatures)
    if y2 == y_leaf:
        tree[bestFeatureLabel][y1] = {}
        tree[bestFeatureLabel][y1] = createTree(datasetNew, subFeatures)
    return tree

# 使用matplotlib绘制树形图
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrows_args)

# 在父子节点间填充文本信息
def plotMidText(cntrPt, parentPt, textString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, textString, va='center', ha='center', rotation=30)

def plotTree(myTree, parentPt, nodeText):
    # 求出宽和高
    numLeafs, depth = getNumLeafs(myTree), getTreeDepth(myTree)
    firstSlides = list(myTree.keys())
    firstStr = firstSlides[0]
    # 按照叶子节点个数划分x轴
    cntrPt = (plotTree.xOff + (0.1 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeText)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # y方向上的摆放位置 自上而下绘制，因此递减y值
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW  # x方向计算结点坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  # 绘制
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  # 添加文本信息
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD  # 下次重新调用时恢复y

# 获取叶子节点数目
def getNumLeafs(myTree):
    # 初始化结点数
    numLeafs = 0
    firstSides = list(myTree.keys())
    # 找到输入的第一个元素，第一个关键词为划分数据集类别的标签
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    # print(secondDict)
    for key in secondDict.keys():
        # 测试节点数据是否为字典
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstSlides = list(myTree.keys())
    firstStr = firstSlides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def createPlot(inTree):
    # 创建一个新图形并清空绘图区
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    createPlot.ax1.axis('off')
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # plt.show()
    plt.savefig('static/images/cart.png')
    
if __name__ == '__main__':
    dataset, features = createDataset()
    myTree = createTree(dataset, features)
    createPlot(myTree)