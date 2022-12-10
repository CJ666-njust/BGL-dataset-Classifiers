import pandas as pd
import numpy as np
import csv

# 测试代码思路用
if __name__ == '__main__':
    embed_dir = 'data_pre/content_embd_2k.csv'
    embed_data = []
    with open(embed_dir, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) > 4:
                line = list(map(float, line))
                embed_data.append(line)
                print(line)
