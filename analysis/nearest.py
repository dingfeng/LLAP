# -*- coding: UTF-8 -*-
# filename: svm date: 2018/12/13 11:26  
# author: FD 
import os
import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from dtw import dtw
from scipy.linalg import norm


def main():
    data = readdata()
    chenhao_data = data['zhuyan']
    dengyufeng_data = data['zhangqian']
    chenhao_dataset = []
    dengyufeng_dataset = []
    for i in range(len(chenhao_data)):
        I = chenhao_data[i]['I']
        chenhao_dataset.append(I)
    for i in range(len(dengyufeng_data)):
        I = dengyufeng_data[i]['I']
        dengyufeng_dataset.append(I)
    chenhao_dataset, chenhao_indexes = shuffle(chenhao_dataset)
    dengyufeng_dataset, dengyufeng_indexes = shuffle(dengyufeng_dataset)
    template_count = 12
    chenhao_template = chenhao_dataset[:template_count]
    dengyufeng_template = dengyufeng_dataset[:template_count]
    right_count = 0
    for i in range(template_count, len(chenhao_dataset)):
        minDistance1 = 10000
        for j in range(template_count):
            distance = get_distance_inlist(chenhao_template[j], chenhao_dataset[i])
            if distance < minDistance1:
                minDistance1 = distance
        minDistance2 = 10000
        for j in range(template_count):
            distance = get_distance_inlist(dengyufeng_template[j], chenhao_dataset[i])
            if distance < minDistance2:
                minDistance2 = distance
        print('index {} min1 min2 {} {} '.format(chenhao_indexes[i], minDistance1, minDistance2))
        if minDistance1 < minDistance2:
            right_count += 1
            print('right')
        else:
            print('wrong')
    print('accuracy {}'.format(right_count / (len(chenhao_dataset) - template_count)))

    pass


def shuffle(data):
    indexes = np.arange(len(data))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data[index])
    return result, indexes


def get_distance_inlist(actionlist0, actionlist1):
    min_distance = 10000
    dctlist0 = []
    for i in range(len(actionlist0)):
        data = actionlist0[i]
        data = normalize(data)
        dctlist0.append(dct(data)[:40])
    dctlist1 = []
    for i in range(len(actionlist1)):
        data = actionlist1[i]
        # data = data - data[0]
        data = normalize(data)
        dctlist1.append(dct(data)[:40])
    for i in range(len(dctlist0)):
        for j in range(len(dctlist1)):
            distance = get_distance(dctlist0[i], dctlist1[j])
            if distance < min_distance:
                min_distance = distance
    return min_distance


def get_distance(data0, data1):
    data0 = data0.reshape(-1, 1)
    data1 = data1.reshape(-1, 1)
    dist, cost, acc, path = dtw(data0, data1, dist=lambda x, y: norm(x - y, ord=1))
    return dist


def readdata():
    dir_path = '../dataset/data20-10/cutted'
    filenames = os.listdir(dir_path)
    data = {}
    for dirname in filenames:
        data_dir = os.path.join(dir_path, dirname)
        data_dir_data = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            onedata = np.load(open(filepath, 'rb'))
            data_dir_data.append(onedata)
        data[dirname] = data_dir_data
    return data


def normalize(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


if __name__ == '__main__':
    # main() dct 每个维取最小值
    main()
