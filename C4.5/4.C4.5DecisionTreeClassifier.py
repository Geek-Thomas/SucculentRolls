#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: GeekThomas
# time: 2021/7/6
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义文本框和箭头格式（树节点格式的常量）
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrows_args = dict(arrowstyle='<-')


# 导入数据集
def createDataset():
    """
    :return: dataSet是二维列表，其中第三列为分类标签 features是特征列表
    """
    df = pd.read_csv('example_data.csv')
    dataSet = df.values.tolist()
    features = df.columns.tolist()[:-1]
    return dataSet, features

# 计算信息熵
def calEntropy(dataset):
    nums = len(dataset)  # 数据总条数
    labels = {}     # 统计数据中各类别数目
    for featureVec in dataset:
        # 每行数据最后一个值为标签
        currentLabel = featureVec[-1]
        labels[currentLabel] = labels.get(currentLabel, 0) + 1
    entropy = 0.0

    for label in labels:
        prob = labels[label] / nums     # 计算每个类别出现概率
        entropy -= prob * np.log2(prob) # 计算信息熵
    return entropy

# 划分数据集
def splitDataset(dataset, col, value):
    """
    :param dataset: 
    :param col: 列序号
    :param value: 对应列的特征值
    :return: 根据该特征值划分后的数据集（删除特征所在列，取特征值=value的行）
    """
    split_result = []
    for featureVec in dataset:
        if featureVec[col] == value:
            subDataset1 = featureVec[:col] + featureVec[col + 1:]
            split_result.append(subDataset1)
    return split_result

# 选择最优特征
def chooseBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1   # 特征数量
    baseEntropy = calEntropy(dataset)   # 信息熵
    bestEntropyGain, bestFeature = 0, -1
    for i in range(numFeatures):
        # 当前特征列下的所有值
        featureValues = [featureVec[i] for featureVec in dataset]
        # print(featureValues)
        # 特征值类别
        featureValueUniques = list(set(featureValues))
        newEntropy = 0.0
        splitInfo = 0.0
        for featureValue in featureValueUniques:
            split_result = splitDataset(dataset, i, featureValue)
            # 求出该值在第i列中出现概率
            prob = len(split_result) / len(dataset)
            # 求第i列特征各值对应的熵之和
            newEntropy += prob * calEntropy(split_result)
            splitInfo = -prob * np.log2(prob)
        # 求出第i列特征的信息增益率
        infoGain = (baseEntropy - newEntropy) / splitInfo
        # 贪心算法获得最大信息增益率对应的特征
        if infoGain > bestEntropyGain:
            bestEntropyGain, bestFeature = infoGain, i
    return bestFeature

# 投票法
def majority(labels):
    return Counter(labels).most_common()[0][0]


def createDecisionTree(dataset, features):
    # 1.如果数据集中所有数据属于同一类
    labels = [featureVec[-1] for featureVec in dataset]
    if len(set(labels)) == 1:
        return labels[0]
    # 2.如果数据集的特征列为空，即只有标签列，则根据投票法返回数目最多的类别
    if len(dataset[0]) == 1:
        return majority(labels)
    # 3.否则，就计算每个特征的信息增益，选出最优特征
    bestFeature = chooseBestFeature(dataset)    # 这是下标
    bestFeatureValue = features[bestFeature]    # 这是最优特征
    # 以最优特征为根节点创建树
    myTree = {bestFeatureValue: {}}
    # 删除掉最优特征
    del features[bestFeature]
    # 找出该特征所有训练数据的值
    featureValues = [featureVec[bestFeature] for featureVec in dataset]
    featureUniqueValues = list(set(featureValues))
    #根据该属性的值求树的各个分支
    for featureVal in featureUniqueValues:
        subFeatures = features[:]
        # print(splitDataset(dataset, bestFeature, bestFeatureValue))
        myTree[bestFeatureValue][featureVal] = createDecisionTree(splitDataset(dataset, bestFeature, featureVal), subFeatures)

    return myTree

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
    plt.show()

if __name__ == '__main__':
    dataset, features = createDataset()
    myTree = createDecisionTree(dataset, features)
    createPlot(myTree)