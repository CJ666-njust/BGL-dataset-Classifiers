import random
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pre.data_check import Datapreprocessor

# 多层感知机的简单实现
class MlpModel(nn.Module):

    def __init__(self, in_channel, out_channel):    # 初始化时，确定输入维度和输出维度
        super(MlpModel, self).__init__()

        self.linear1 = nn.Linear(in_channel, 256)   # 全连接层1
        self.relu = nn.ReLU()                       # 激活函数
        self.linear2 = nn.Linear(256, 256)          # 全连接层2
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, out_channel)  # 全连接层3

    # 前向传播函数
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


model = MlpModel(768, 380)
# loss
loss = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


def run_mlp(root_dir, embed_dir):
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
        model = MLPClassifier(solver='lbfgs', activation='logistic').fit(trainStd, train_label)

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
    run_mlp(root_dir, embed_dir)

