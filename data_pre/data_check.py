import os

import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Datapreprocessor(object):

    """对BGL数据集的标准格式进行预处理。"""
    def __init__(self, data_dir: str, embed_dir: str):
        self.data_dir = data_dir
        self.embed_dir = embed_dir
        self.required_columns = ['Label', 'Component', 'Content', 'EventId']    # 需要的与分类结果有关的数据(col idx:1,8,10,11)
        self.data, self.label = self.get_data()
        self.class_dict = self.count_events()       # 标签字典集
        self.embed_data = self.get_embed_data()     # 编码后的数据
        self.label_to_int()

        if len(self.embed_data) == len(self.label):
            print("data len:", len(self.embed_data))
            print("data init finished.")
        else:
            print("the len of data and label is not equal.")
            print("embed data len:", len(self.embed_data))
            print("label len：", len(self.label))

    # 获取数据(完成第一次数据分析后，不需要读取其他数据，只需要标签,节约资源)
    def get_data(self):
        assert os.path.exists(self.data_dir), "data path {} does not exist.".format(self.data_dir)

        print("loading necessary data and label...")
        data = pd.read_csv(self.data_dir, nrows=100000)  # 仅读取前100000行

        # new_data = data[[col for col in self.required_columns]]  # 仅保留需要的列
        label = data[[col for col in ['EventId']]]

        # data = new_data.values.tolist()
        label = label.values.tolist()
        label = [i for item in label for i in item]  # 展平为一维数组
        # print(label)
        data = None
        return data, label

    # 获取编码后的content数据
    def get_embed_data(self):
        assert os.path.exists(self.embed_dir), "data path {} does not exist.".format(self.embed_dir)
        embed_data = []
        i = 0
        print("loading embed data: 'Content'...")
        with open(self.embed_dir, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if i == 100000:  # 共处理100000条数据
                    break
                if len(line) > 4:   # 剔除[]行
                    line = list(map(float, line))   # 将str列表转换为float列表
                    embed_data.append(line)
                    i += 1
        return embed_data

    # 获取类别信息的分类编码字典,格式(key, value) = (Ei, i).# BGL2k数据集共120种,从E1到E120; BGL500m数据集共380种
    def count_events(self):
        event_list = []
        for label in self.label:
            if label not in event_list:
                event_list.append(label)
        # event_list.sort()
        # print("event list:", event_list)
        # 建立类别字典集
        event_dict = dict((value, int(value.split('E')[1])) for value in event_list)
        # print("event_dict:", event_dict)
        # print(len(event_dict))
        return event_dict

    # 将标签从字符串转为int
    def label_to_int(self):
        self.label = [self.class_dict[i] for i in self.label]
        # print(self.label)

    # 数据分析：统计其余信息的分类编码的总数
    def one_hot_info(self):
        Label_list = []
        Component_list = []
        Content_list = []
        for info in self.data:
            if info[0] not in Label_list:
                Label_list.append(info[0])
            if info[1] not in Component_list:
                Component_list.append(info[1])
            if info[2] not in Content_list:
                Content_list.append(info[2])

        print("Label len:", len(Label_list))
        print("Component len:", len(Component_list))
        print("Content len:", len(Content_list))


if __name__ == '__main__':

    root_dir = r"../2k_data/BGL/BGL_2k.log_structured.csv"
    # root_dir = r"../../data/BGL/BGL_500m.log_structured.csv"
    embed_dir = 'content_embd_2k.csv'
    data_preprocessor = Datapreprocessor(data_dir=root_dir, embed_dir=embed_dir)
    # data_preprocessor.one_hot_info()
