import random
import time

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pre.data_check import Datapreprocessor

# knn算法的简单实现
def knn(Pred_data, dataSet, labels, k=5):
    global ans
    rows = dataSet.shape[0]  # 计算有多少组特征值

    # 计算待预测样本与训练数据集中样本特征之间的欧式距离
    diff = np.tile(Pred_data, (rows, 1)) - dataSet  # 将pred_data重复rows次
    sqrt_dist = np.sum(diff**2, axis=1)  # 按行相加，不保持其二维特性
    distance = sqrt_dist ** 0.5  # 开方

    # 按照距离递增的顺序排序
    sorted_indices = np.argsort(distance)

    # 选取距离最近的K个样本以及所属类别的次数
    map_label = {}
    for i in range(k):
        label = labels[sorted_indices[i]][0]
        map_label[label] = map_label.get(label, 0) + 1

    # 返回前k个点所出现频率最高的类别作为预测分类结果
    max_num = 0
    for key, value in map_label.items():
        if value > max_num:
            max_num = value
            ans = key
    return ans


def run_knn(root_dir, embed_dir):
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
        model = KNeighborsClassifier().fit(trainStd, train_label)

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
    embed_dir = '../data_pre/content_embd_2k.csv'
    # embed_dir = 'content_embd_500m.csv'
    run_knn(root_dir, embed_dir)


