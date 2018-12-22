# -*- coding: UTF-8 -*-
# filename: show_imag date: 2018/12/19 21:29  
# author: FD 
import numpy as np
from scipy.signal import butter, lfilter, find_peaks_cwt
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import norm
import pandas
from analysis.utils import plot_fft
from scipy.stats import pearsonr

fs = 48000
freq = 20000


def main():
    dir_path = 'raw/dingfeng'
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        image_name = ''.join(['image/', filename[:4], '.png'])
        generate_photo(filepath, image_name)
        # break


def generate_photo(file_path, image_name):
    data = np.memmap(file_path, dtype=np.float32, mode='r')
    previous = None
    start_pos = None
    end_pos = None
    cutted_IQs = []
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
            plt.figure()
            plt.scatter([i for i in range(len(IQ))], IQ)
            plt.plot([start_pos, start_pos], [np.min(IQ), np.max(IQ)], c='r')
            plt.plot([end_pos, end_pos], [np.min(IQ), np.max(IQ)], c='r')
            print('variance[ {} ] = {}'.format(i, np.var(IQ[start_pos:end_pos])))
            plt.ylim(-0.002, 0.002)
            plt.title(file_path)
            points = plt.ginput(5, timeout=0)
            # if len(points) > 0:
        cutted_IQ = IQ[start_pos:end_pos]
        previous = cutted_IQ
        # plt.show()


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
    if (len(long_data) < len(short_data)):
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
