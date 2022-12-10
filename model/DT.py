import random
import time

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pre.data_check import Datapreprocessor


# 信息熵的计算
def calEnt(dataSet):
    n = dataSet.shape[0]  # 数据集总行数
    iset = dataSet.iloc[:, -1].value_counts()  # 标签的所有类别
    p = iset / n  # 每一类标签所占比
    ent = (-p * np.log2(p)).sum()  # 计算信息熵
    return ent

# 选择最大信息熵的列进行切分
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)  # 计算原始熵
    bestGain = 0  # 初始化信息增益
    axis = -1  # 初始化最佳切分列，标签列
    for i in range(dataSet.shape[1] - 1):  # 对特征的每一列进行循环
        levels = dataSet.iloc[:, i].value_counts().index  # 提取出当前列的所有取值
        ents = 0  # 初始化子节点的信息熵
        for j in levels:  # 对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:, i] == j]  # 某一个子节点的dataframe
            ent = calEnt(childSet)  # 计算某一个子节点的信息熵
            ents += (childSet.shape[0] / dataSet.shape[0]) * ent  # 计算当前列的信息熵
        # print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt - ents  # 计算当前列的信息增益
        # print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain  # 选择最大信息增益
            axis = i  # 最大信息增益所在列的索引
    return axis

# 按索引切分数据
def mySplit(dataSet, axis, value):
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return redataSet

# 创建决策树
def createTree(dataSet):
    featlist = list(dataSet.columns)  # 提取出数据集所有的列
    classlist = dataSet.iloc[:, -1].value_counts()  # 获取最后一列类标签
    # 判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]  # 如果是，返回类标签
    axis = bestSplit(dataSet)  # 确定出当前最佳切分列的索引
    bestfeat = featlist[axis]  # 获取该索引对应的特征
    myTree = {bestfeat: {}}  # 采用字典嵌套的方式存储树信息
    del featlist[axis]  # 删除当前特征
    valuelist = set(dataSet.iloc[:, axis])  # 提取最佳切分列所有属性值
    for value in valuelist:  # 对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet, axis, value))
    return myTree

# 使用决策树进行分类
def classify(inputTree, labels, testVec):
    firstStr = next(iter(inputTree))  # 获取决策树第一个节点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = labels.index(firstStr)  # 第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 决策树的分类预测
def acc_classify(train, test):
    inputTree = createTree(train)  # 根据测试集生成一棵树
    labels = list(train.columns)  # 数据集所有的列名称
    result = []
    for i in range(test.shape[0]):  # 对测试集中每一条数据进行循环
        testVec = test.iloc[i, :-1]  # 测试集中的一个实例
        classLabel = classify(inputTree, labels, testVec)  # 预测该实例的分类
        result.append(classLabel)  # 将分类结果追加到result列表中
    test['predict'] = result  # 将预测结果追加到测试集最后一列
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()  # 计算准确率
    print(f'模型预测准确率为{acc}')
    return test



def run_dt(root_dir, embed_dir):
    random.seed(0)  # 保证随机结果可复现
    # 数据预处理
    data_preprocessor = Datapreprocessor(data_dir=root_dir, embed_dir=embed_dir)
    X = data_preprocessor.embed_data
    Y = data_preprocessor.label

    avg_acc = 0
    avg_time = 0

    # 十折交叉验证
    for i in range(0, 10):
        print("the {} epoch.".format(i + 1))
        # 数据集划分和建立模型
        train_data, test_data, train_label, test_label = \
            train_test_split(X, Y, test_size=0.2)

        # 数据标准化
        std = StandardScaler().fit(train_data)
        trainStd = std.transform(train_data)
        testStd = std.transform(test_data)

        # 训练，并计算时间差
        a = time.time()
        model = DecisionTreeClassifier(criterion="entropy").fit(trainStd, train_label)

        # 预测并给出准确率
        y_ = model.predict(testStd)
        acc = np.sum(y_ == test_label) / len(test_label)
        print("预测正确数量：", np.sum(y_ == test_label))  # 391
        print("模型准确率：", acc)  # 0.9775

        b = time.time() - a
        print("训练+预测时间:{:.4f}s.".format(b))

        avg_acc += acc
        avg_time += b

    avg_acc /= 10
    avg_time /= 10

    print("平均准确率：{:.4f}。".format(avg_acc))
    print("平均 （数据划分+训练+预测） 时间：{:.4f}s。".format(avg_time))


if __name__ == '__main__':
    root_dir = r"../2k_data/BGL/BGL_2k.log_structured.csv"
    # root_dir = r"../data/BGL/BGL_500m.log_structured.csv"
    embed_dir = '../../data_pre/content_embd_2k.csv'
    # embed_dir = 'content_embd_500m.csv'
    run_dt(root_dir, embed_dir)
