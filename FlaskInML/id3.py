#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: GeekThomas
# time: 2021/7/1
import pandas as pd
import matplotlib.pyplot as plt
from math import log
import os

# 定义文本框和箭头格式（树节点格式的常量）
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrows_args = dict(arrowstyle='<-')

# 加载数据集
def createDataset():
    """
    :return: dataSet是二维列表，其中第三列为分类标签 features是特征列表
    """
    df = pd.read_csv('example_data.csv')
    dataSet = df.values.tolist()
    features = df.columns.tolist()
    return dataSet, features

# 计算数据集的信息熵
def calEntropy(dataSet):
    """
    :param dataSet: 传入数据集
    :return: 返回信息熵：0.9402859586706309
    """
    # 统计数据集个数
    nums = len(dataSet)
    labelCount = {}
    for featureVec in dataSet:
        # 每行数据最后一个值为所属类别
        currentLabel = featureVec[-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1

    # 计算信息熵
    entropy = 0.0
    for label in labelCount:
        prob = float(labelCount[label]) / nums
        entropy += -prob * log(prob, 2)
    return entropy

# 划分数据集
def splitDataSet(dataSet, featureCol, feature):
    """
    :param dataSet: 数据集dataSet
    :param featureCol: 划分特征所在的列序号
    :param feature: 特征对应值
    :return:
    """
    split_result = []
    for dataVec in dataSet:
        if dataVec[featureCol] == feature:
            reduceFeaVec = dataVec[:featureCol]
            reduceFeaVec.extend(dataVec[featureCol+1:])
            split_result.append(reduceFeaVec)
    return split_result

# 选择最好的划分方法-只需要得到对应的划分特征
def chooseBestSplitMethod(dataSet):
    """
    :param dataSet: 传入数据集变量
    :return: 返回最佳划分特征对应的信息增益以及对应的列顺序
    """
    # 对每个特征进行遍历：这里对应的列顺序
    featureCols = list(range(len(dataSet[0]) - 1))
    baseEntropy = calEntropy(dataSet)
    bestEntropyGain, bestFeature = 0, -1
    for i in featureCols:
        # 记录下当前特征列的所有值
        featureValues = [dataVec[i] for dataVec in dataSet]
        featureValueUniques = list(set(featureValues))
        newEntropy = 0.0
        # 根据当前列以及特征列值划分数据集
        for featureValue in featureValueUniques:
            # print(feature)
            split_result = splitDataSet(dataSet, i, featureValue)
            # print(split_result)
            # 计算信息熵
            prob = float(len(split_result)) / len(dataSet)
            newEntropy += prob * calEntropy(split_result)
        # 计算信息增益
        entropyGain = baseEntropy - newEntropy
        # print(entropyGain)
        # 贪心算法
        if entropyGain > bestEntropyGain:
            bestEntropyGain, bestFeature = entropyGain, i
    return bestFeature

# 使用投票法，当特征全部划分完，仍然无法分类的，根据分类多数赋值
def majority(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createDecisionTree(dataSet, features):
    # 包含数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，则停止划分
    if len(set(classList)) == 1:
        return classList[0]
    # 当遍历完所有特征，但是仍然不能将数据集划分成仅包含唯一分类的分组
    # 返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majority(classList)
    # 接下来就是递归：得到最优特征
    # 这个得到的是序号
    currentBestFeature = chooseBestSplitMethod(dataSet)
    # print(currentBestFeature)
    currentBestFeatureLabel = features[currentBestFeature]
    myTree = {currentBestFeatureLabel: {}}
    # 将最优特征剔除
    del(features[currentBestFeature])
    # 得到列表包含的属性值
    featureValues = [dataVec[currentBestFeature] for dataVec in dataSet]
    featureValueUniques = set(featureValues)

    for featureValueUnique in featureValueUniques:
        subFeatures = features[:]
        myTree[currentBestFeatureLabel][featureValueUnique] = createDecisionTree(splitDataSet(dataSet, currentBestFeature, featureValueUnique), subFeatures)
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
    # plt.show()
    plt.savefig("static/images/id3.png")

if __name__ == '__main__':
    dataSet, features = createDataset()
    # print(splitDataSet(dataSet, 0, "high"))
    myTree = createDecisionTree(dataSet, features)
    createPlot(myTree)
