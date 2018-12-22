# -*- coding: UTF-8 -*-
# filename: cut_data date: 2018/12/22 13:33  
# author: FD
import numpy as np
from scipy.signal import butter, lfilter, find_peaks_cwt
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import norm
import pandas
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids

fs = 48000
freq = 20000
NUM_CLUSTERS = 5


def main():
    cut_dir('../dataset/dingfeng_big_write/raw/yuhuan', '../dataset/dingfeng_big_write/cutted/yuhuan')
    return


def cut_dir(source_dir, dest_dir):
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('pcm'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, ''.join([filename[:-4], '.pkl']))
            cut_file(source_path, dest_path)




def cut_file(source_filepath, dest_filepath):
    global cluster_count
    data = np.memmap(source_filepath, dtype=np.float32, mode='r')
    previous = None
    start_pos = None
    end_pos = None
    cutted_IQs = []
    previous_added = False
    for i in range(64):
        I = getI(data, freq, i * np.pi / 32)
        I = move_average(I)
        Q = getQ(data, freq, i * np.pi / 32)
        Q = move_average(Q)
        decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
        I = decompositionI.trend[10:]
        decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
        Q = decompositionQ.trend[10:]
        IQ = np.asarray([I, Q]).T
        IQ = IQ - np.roll(IQ, 1)
        IQ = IQ[1:]
        IQ = norm(IQ, ord=2, axis=1)
        IQ = IQ - np.roll(IQ, 1)
        IQ = IQ[1:]
        if previous is None:
            start_pos, end_pos = get_bounds(IQ)
            fig = plt.figure()
            plt.scatter([i for i in range(len(IQ))], IQ)
            plt.plot([start_pos, start_pos], [np.min(IQ), np.max(IQ)], c='r')
            plt.plot([end_pos, end_pos], [np.min(IQ), np.max(IQ)], c='r')
            print('variance[ {} ] = {}'.format(i, np.var(IQ[start_pos:end_pos])))
            plt.ylim(-0.002, 0.002)
            plt.title(source_filepath)
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            points = plt.ginput(5, timeout=0)
            plt.close(fig)
            if len(points) > 0:
                start_pos = int(points[0][0])
                end_pos = int(points[1][0])
            # print(points)
        cutted_IQ = IQ[start_pos:end_pos]
        if previous is not None:
            correlation = get_correlation(cutted_IQ, previous)
            if correlation > 0.95 and previous is not None:
                cutted_IQs.append(cutted_IQ)
                if not previous_added:
                    cutted_IQs.append(previous)
                    previous_added = True
            else:
                previous_added = False
        previous = cutted_IQ
    # 使用方差筛选去除最小的3条曲线
    vars = []
    for cutted_IQ in cutted_IQs:
        vars.append(np.var(cutted_IQ))
    new_cutted_IQs = []
    for index in np.argsort(vars)[3:]:
        new_cutted_IQs.append(cutted_IQs[index])
    distances_mat = np.zeros((len(new_cutted_IQs), len(new_cutted_IQs)))
    for i in range(len(new_cutted_IQs)):
        for j in range(i + 1, len(new_cutted_IQs)):
            distances_mat[i, j] = distances_mat[j, i] = 1 - get_correlation(new_cutted_IQs[i], new_cutted_IQs[j])
    indexes = np.arange(len(new_cutted_IQs))
    np.random.shuffle(indexes)
    kmedoids_instance = kmedoids(distances_mat, indexes[:NUM_CLUSTERS], data_type='distance_matrix')
    kmedoids_instance.process()
    medoids = kmedoids_instance.get_medoids()
    final_features = []
    for medoid in medoids:
        final_features.append(new_cutted_IQs[medoid])
    pickle.dump(final_features, open(dest_filepath, 'wb'))

def get_PAM_distance(index0, index1):
    global distances_mat
    print('index 0 {} index 1 {}'.format(index0, index1))
    return distances_mat[index0, index1]


def get_bounds(data):
    ps = pandas.Series(data=data)
    var = ps.rolling(window=7).var()
    var[:6] = 0
    threshold = 0.5e-8
    search_start = 60
    # find the start pos of series
    start_pos = None
    for i in range(search_start, len(var)):
        if var[i] > threshold:
            start_pos = i
            start_pos -= 60
            break
    # find the end pos of series
    end_pos = None
    for i in range(len(var)):
        end_pos = len(var) - 1 - i
        if var[end_pos] > threshold:
            end_pos += 40
            break
    # plt.figure()
    # plt.plot(var)
    return start_pos, end_pos


def getI(data, f, biase):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times + biase) * data
    return mulCos


def getQ(data, f, biase):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times + biase) * data
    return mulSin


def get_correlation(data0, data1):
    short_data = data0
    long_data = data1
    if len(long_data) < len(short_data):
        temp = short_data
        short_data = long_data
        long_data = temp
    lags = [i for i in range(len(long_data) - len(short_data) + 1)]
    max_pearson = -2
    short_data_len = len(short_data)
    for lag in lags:
        pearson_value = pearsonr(long_data[lag:lag + short_data_len], short_data)[0]
        max_pearson = max(max_pearson, pearson_value)
    return max_pearson


def move_average(data):
    win_size = 300
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result


if __name__ == '__main__':
    main()
