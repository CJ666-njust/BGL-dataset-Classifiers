import random
import time

import numpy as np
import sklearn.svm as svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pre.data_check import Datapreprocessor

# 线性SVM的简单实现
class LinearSVM(object):
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, c=1, lr=0.01, epoch=1000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        for _ in range(epoch):  # 训练迭代
            self._w *= 1 - lr   # 按照学习率等比更新参数
            err = 1 - y * self.predict(x, True)
            idx = np.argmax(err)

            if err[idx] <= 0:   # 若样本被正确分类，继续迭代
                continue
            delta = lr * c * y[idx]     # 按照负梯度方向更新参数
            self._w += delta * x[idx]
            self._b += delta

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self._w) + self._b   # 计算 w·x+b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)


def run_svm(root_dir, embed_dir):
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
        model = svm.SVC(kernel="linear", decision_function_shape="ovo").fit(trainStd, train_label)

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
    run_svm(root_dir, embed_dir)
