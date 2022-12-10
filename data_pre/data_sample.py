import pandas as pd
import numpy as np
import csv


# 读入数据
required_columns = ['Content', 'EventId']
data_dir = r"../../data/BGL/BGL_500m.log_structured.csv"
# data_dir = r"../2k_data/BGL/BGL_2k.log_structured.csv"

write_dir = r"BGL_100k.csv"
csv_dir = r"content_embd_100k.csv"
data = pd.read_csv(data_dir)
data = data[[col for col in required_columns]]
# 随机采样100k个样本
data = data.sample(n=100000, replace=False, random_state=1)
data.to_csv(write_dir)
