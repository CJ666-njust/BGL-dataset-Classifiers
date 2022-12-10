import pandas as pd
import numpy as np
import csv

from bert_serving.client import BertClient  # 导入客户端

# 此处使用bert-serving对单词进行编码，使用的python环境:python==3.6.12,tensorflow==1.15.0,bert_serving

# 读入数据
required_columns = ['Content']
data_dir = r"BGL_100k.csv"
# data_dir = r"../2k_data/BGL/BGL_2k.log_structured.csv"

csv_dir = r"content_embd_100k.csv"
data = pd.read_csv(data_dir)
# data.sample(n=100000, replace=False, random_state=1)
# data.to_csv(write_dir)

new_data = data[[col for col in required_columns]]  # 仅保留需要的列
data = new_data.values.tolist()
print("data load finish.")
# 创建客户端对象
bc = BertClient()
# result_list = []
print("begin.")

np.set_printoptions(suppress=True)  # 不使用科学计数法

with open(csv_dir, "w") as f:
    writer = csv.writer(f)

    for i, content in enumerate(data):
        # if i == 5:
        #     break
        if i % 10000 == 1:
            print("Bert embed process:{}%.".format(i/1000))

        writer.writerow(np.around(bc.encode(content)[0], decimals=4))


