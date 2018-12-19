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
from dtw import dtw
from scipy.spatial.distance import euclidean
from scipy.fftpack import dct, fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from scipy.stats import pearsonr

start_freq = 0
end_freq = 35


def main():
    data0 = get_data('../dataset/data20-10/cutted_IQ/anna/')
    data1 = get_data('../dataset/data20-10/cutted_IQ/yingjunhao/')
    template_count = 10
    template0 = data0[:template_count]
    template1 = data1[:template_count]
    compares = data0[template_count:]
    count = 0
    for compared in compares:
        min_distance0 = get_distance_to_template(template0, compared)
        min_distance1 = get_distance_to_template(template1, compared)
        print('min_distance-0 {} min_distance-1 {}'.format(min_distance0, min_distance1))
        if min_distance0 < min_distance1:
            print('right')
            count += 1
        else:
            print('wrong')
    print('right {} total {} accuracy {}'.format(count, len(compares), count / len(compares)))


def get_distance_to_template(templates, data):
    min_distance = 10000
    for template in templates:
        for template_feature in template:
            distance = get_distance(template_feature, data[0])  # get_distance(template, data)
        min_distance = min(min_distance, distance)
    return min_distance


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        datas = np.load(open(filepath, 'rb'))
        label_data = []
        for data in datas:
            data = data - np.roll(data, 1)
            data = data[1:]
            data = norm(data, ord=2, axis=1)
            # data = data - np.roll(data, 1)
            # data = data[1:]
            # data = data - np.roll(data, 1)
            # data = data[1:]
            # data = np.abs(data)
            data = reshape_data(data)
            ss1 = MinMaxScaler()
            ss1.fit(data)
            data = ss1.transform(data)
            feature = dct(data, axis=0)[start_freq:end_freq]
            label_data.append(feature)
        result.append(label_data)
    result = shuffle(result)
    return result


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result


def get_distance(feature0, feature1):
    # dist, cost, acc, path = dtw(feature0, feature1, dist=lambda x, y: norm(x - y, ord=1))
    # print('dtw')
    return norm(feature0-feature1,ord=2)  # pearsonr(feature0,feature1)[0]


def reshape_data(data):
    data = normalize(data)
    next_data = np.zeros(1800)
    next_data[50:len(data) + 50] = data[:]
    next_data = next_data.reshape(1800, 1)
    return next_data


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
