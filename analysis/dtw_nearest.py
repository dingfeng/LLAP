# -*- coding: UTF-8 -*-
# filename: dtw_nearest date: 2018/12/18 18:16  
# author: FD 
import os
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    data0 = get_data('../dataset/data20-10/max_variance_cutted_features/huangsi/')
    data1 = get_data('../dataset/data20-10/max_variance_cutted_features/zhuyan/')


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))
        result.append(data[0])
    result = shuffle(result)
    return result


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result


if __name__ == '__main__':
    main()
