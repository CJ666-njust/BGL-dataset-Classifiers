import random
import time
import csv

import numpy as np

from random import randrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pre.data_check import Datapreprocessor

# 加载数据，一行行的存入列表
def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

# 除了标签列，其他列都转换为float类型
def column_to_float(dataSet):
    featLen = len(dataSet[0]) - 1
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())

# 将数据集随机分成N块，方便交叉验证，其中一块是测试集，其他四块是训练集
def spiltDataSet(dataSet, n_folds):
    fold_size = int(len(dataSet) / n_folds)
    dataSet_copy = list(dataSet)
    dataSet_spilt = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:  # 这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        dataSet_spilt.append(fold)
    return dataSet_spilt

# 构造数据子集
def get_subsample(dataSet, ratio):
    subdataSet = []
    lenSubdata = round(len(dataSet) * ratio)  # 返回浮点数
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet) - 1)
        subdataSet.append(dataSet[index])
    # print len(subdataSet)
    return subdataSet

# 分割数据集
def data_spilt(dataSet, index, value):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# 计算分割代价
def spilt_loss(left, right, class_values):
    loss = 0.0
    for class_value in class_values:
        left_size = len(left)
        if left_size != 0:  # 防止除数为零
            prop = [row[-1] for row in left].count(class_value) / float(left_size)
            loss += (prop * (1.0 - prop))
        right_size = len(right)
        if right_size != 0:
            prop = [row[-1] for row in right].count(class_value) / float(right_size)
            loss += (prop * (1.0 - prop))
    return loss

# 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_spilt(dataSet, n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
    while len(features) < n_features:
        index = randrange(len(dataSet[0]) - 1)
        if index not in features:
            features.append(index)
    # print 'features:',features
    for index in features:  # 找到列的最适合做节点的索引，（损失最小）
        for row in dataSet:
            left, right = data_spilt(dataSet, index, row[index])  # 以它为节点的，左右分支
            loss = spilt_loss(left, right, class_values)
            if loss < b_loss:  # 寻找最小分割代价
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
    # print b_loss
    # print type(b_index)
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}

# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]
    return max(set(output), key=output.count)

# 子分割，不断地构建叶节点的过程对对对
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']
    # print left
    right = root['right']
    del (root['left'])
    del (root['right'])
    # print depth
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)
        # print 'testing'
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        root['left'] = get_best_spilt(left, n_features)
        # print 'testing_left'
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right, n_features)
        # print 'testing_right'
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)

# 构造决策树
def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)
    sub_spilt(root, n_features, max_depth, min_size, 1)
    return root

# 预测测试集结果
def predict(tree, row):
    predictions = []
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']
            # predictions=set(predictions)

# 批量预测
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# 创建随机森林
def random_forest(train, test, ratio, n_feature, max_depth, min_size, n_trees=10):
    trees = []
    for i in range(n_trees):
        train = get_subsample(train, ratio)  # 从切割的数据集中选取子集
        tree = build_tree(train, n_feature, max_depth, min_size)    # 构造决策树
        # print 'tree %d: '%i,tree
        trees.append(tree)  # 加入森林
    # predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]  # 进行预测
    return predict_values

# 计算准确率
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict_values[i]:
            correct += 1
    return correct / float(len(actual))


def run_rf(root_dir, embed_dir):
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
        model = RandomForestClassifier(n_estimators=10).fit(trainStd, train_label)

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
    run_rf(root_dir, embed_dir)
