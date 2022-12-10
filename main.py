import pandas as pd
import numpy as np

from model.SVM import run_svm
from model.KNN import run_knn
from model.DT import run_dt
from model.MLPclassifier import run_mlp
from model.RandomForest import run_rf


if __name__ == '__main__':

    # 使用BGL-2k数据集
    root_dir = r"./2k_data/BGL/BGL_2k.log_structured.csv"
    embed_dir = 'data_pre/content_embd_2k.csv'

    # 使用BGL-100k数据集
    # root_dir = 'data_pre/BGL_100k.csv'
    # embed_dir = 'data_pre/content_embd_100k.csv'

    run_svm(root_dir, embed_dir)
    print("svm classifier finished.")
    print("____" * 30)

    run_knn(root_dir, embed_dir)
    print("knn classifier finished.")
    print("____" * 30)

    run_dt(root_dir, embed_dir)
    print("decision tree classifier finished.")
    print("____" * 30)

    run_mlp(root_dir, embed_dir)
    print("mlp classifier finished.")
    print("____" * 30)

    run_rf(root_dir, embed_dir)
    print("random forest classifier finished.")
    print("____" * 30)