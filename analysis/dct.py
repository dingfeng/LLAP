# -*- coding: UTF-8 -*-
# filename: dct date: 2018/12/11 15:42  
# author: FD 
import numpy as np
import os
from scipy.fftpack import dct
from dtw import dtw
from scipy.linalg import norm
import matplotlib.pyplot as plt


def main():
    dingfeng_dir = '../dataset/cutted/dingfeng/'
    dingfeng_data = collect_feature(dingfeng_dir)
    anna_dir = '../dataset/cutted/anna/'
    anna_data = collect_feature(anna_dir)
    dingfeng2_dir = '../dataset/cutted/dingfeng2/'
    dingfeng2_data = collect_feature(dingfeng2_dir)
    dist_map = np.zeros((len(anna_data) + len(dingfeng2_data), len(dingfeng_data)))
    # for i in range(len(dingfeng_data)):
    #     for j in range(len(dingfeng_data)):
    #         feature0 = dingfeng_data[i]
    #         feature1 = dingfeng_data[j]
    #         dist = get_distance(feature0, feature1)
    #         dist_map[i, j] = dist
    for i in range(len(anna_data)):
        for j in range(len(dingfeng_data)):
            feature0 = anna_data[i]
            feature1 = dingfeng_data[j]
            dist = get_distance(feature0, feature1)
            dist_map[i, j] = dist

    for i in range(len(anna_data), len(anna_data) + len(dingfeng2_data)):
        for j in range(len(dingfeng_data)):
            feature0 = dingfeng2_data[i -len(anna_data)]
            feature1 = dingfeng_data[j]
            dist = get_distance(feature0, feature1)
            dist_map[i, j] = dist
    result = np.mean(dist_map, axis=1)
    # for i in range(len(dingfeng_data)):
    #     # if(i<len(dingfeng_data)):
    #     result[i] = result[i]# * len(dingfeng_data) / (len(dingfeng_data) - 1)

    for i in np.argsort(result):
        print(i)
    pass


def test():
    dir = '../dataset/cutted/dingfeng'
    dct_array = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        data = np.load(open(filepath, 'rb'))
        I = data['I']
        I = normalize(I)
        Q = data['Q']
        Q = normalize(Q)
        print(filename)
        dct_array.append(np.concatenate((dct(I)[:40], dct(Q)[:40])).reshape(-1, 1))
    first = dct_array[0]
    for i in range(0, len(dct_array)):
        dist, cost, acc, path = dtw(first, dct_array[i], dist=lambda x, y: norm(x - y, ord=1))
        print()


def get_distance(feature0, feature1):
    dist, cost, acc, path = dtw(feature0, feature1, dist=lambda x, y: norm(x - y, ord=1))
    return dist  # / (len(feature0) + len(feature1))


def collect_feature(dir_path):
    dir = dir_path
    dct_array = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        data = np.load(open(filepath, 'rb'))
        I = data['I']
        I = normalize(I)
        Q = data['Q']
        Q = normalize(Q)
        dct_array.append(np.concatenate((dct(I)[:20], dct(Q)[:20])).reshape(-1, 1))
    return dct_array


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
