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
from fastdtw import fastdtw
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    data0 = get_data('../dataset/data20-10/cutted_IQ/dingfeng/')
    data1 = get_data('../dataset/data20-10/cutted_IQ/huangsi/')
    template_count = 10
    template0 = data0[:template_count]
    template1 = data1[:template_count]
    compares = data0[template_count:]
    count = 0

    ss0 = MinMaxScaler()
    ssData = data0[0]
    for i in range(1,len(data0)):
        ssData=np.vstack((ssData,data0[i]))
    ss0.fit(ssData)

    ss1 = MinMaxScaler()
    ssData = data0[0]
    for i in range(1, len(data1)):
        ssData = np.vstack((ssData, data1[i]))
    ss1.fit(ssData)
    dct_template0 = []
    for template in template0:
        dct_template0.append(dct(ss0.transform(template))[:40])
    dct_template1 = []
    for template in template1:
        dct_template1.append(dct(ss1.transform(template))[:40])

    for compared in compares:
        min_distance0 = get_distance_to_template(dct_template0, dct(ss0.transform(compared))[:40])
        min_distance1 = get_distance_to_template(dct_template1, ss1.transform(compared)[:40])
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
        distance = get_distance(template, data)  # get_distance(template, data)
        min_distance = min(min_distance, distance)
    return min_distance


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))[0]
        data = data - np.roll(data, 1)
        data = data[1:]
        data = norm(data, ord=2, axis=1)
        data = data - np.roll(data, 1)
        data = data[1:]
        data = data - np.roll(data, 1)
        data = data[1:]
        # data = np.abs(data)
        data = data.reshape(-1, 1)
        result.append(data)
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
    dist, cost, acc, path = dtw(feature0, feature1, dist=lambda x, y: norm(x - y, ord=1))
    # print('dtw')
    return dist


def get_fast_distance(feature0, feature1):
    distance, path = fastdtw(feature0, feature1)
    return distance


if __name__ == '__main__':
    main()
