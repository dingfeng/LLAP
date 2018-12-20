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
    I = getI(data, freq, 0)
    I = move_average(I)
    Q = getQ(data, freq, 0)
    Q = move_average(Q)
    # plt.plot(I)
    # plt.show()
    # plt.plot(Q)
    # plt.show()
    decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
    # decompositionI.plot()
    # plt.show()
    I = decompositionI.trend[10:]

    decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
    Q = decompositionQ.trend[10:]
    IQ = np.asarray([I, Q]).T
    IQ = IQ - np.roll(IQ, 1)
    IQ = IQ[1:]
    IQ = norm(IQ, ord=2, axis=1)
    IQ = IQ - np.roll(IQ, 1)
    IQ = IQ[1:]
    start_pos, end_pos = get_bounds(IQ)
    plt.figure()
    plt.scatter([i for i in range(len(IQ))], IQ)
    plt.plot([start_pos, start_pos], [np.min(IQ), np.max(IQ)],c='r')
    plt.plot([end_pos, end_pos], [np.min(IQ), np.max(IQ)], c='r')
    # plt.ylim(-0.001,0.001)
    # plt.savefig(image_name)
    plt.show()


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
    #find the end pos of series
    end_pos = None
    for i in range(len(var)):
        end_pos=len(var)-1-i
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
