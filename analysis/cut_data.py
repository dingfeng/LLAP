# -*- coding: UTF-8 -*-
# filename: cut_data date: 2018/12/11 15:13  
# author: FD 
import numpy as np
from scipy.signal import butter, lfilter, find_peaks_cwt
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import norm
fs = 48000
freq = 20000


def main():
    source_dir = '../dataset/data20-10/raw/huangsi'
    dest_dir = '../dataset/data20-10/cutted_IQ/huangsi'
    cut_dir(source_dir, dest_dir)


def cut_dir(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith('pcm'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, ''.join([filename[:-4], '.pkl']))
            cut(source_path, dest_path)


def cut(source_path, dest_path):
    data = np.memmap(source_path, dtype=np.float32, mode='r')
    points = None
    toSaveObj = []
    for i in range(64):
        I = getI(data, freq, i * np.pi / 32)
        I = move_average(I)
        decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
        I = decompositionI.trend
        Q = getQ(data, freq, i * np.pi / 32)
        Q = move_average(Q)
        decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
        Q = decompositionQ.trend
        if i == 0:
            fig = plt.figure()
            plt.subplot(211)
            plt.plot(I)
            plt.subplot(212)
            plt.plot(Q)
            points = plt.ginput(5, timeout=0)
            points_len = len(points)
            print('point length {}'.format(points_len))
            if (points_len == 0):
                plt.close(fig)
                return
            plt.close(fig)
        cutted_I = I[int(points[0][0]):int(points[1][0])]
        cutted_Q = Q[int(points[0][0]):int(points[1][0])]
        cutted_IQ = np.asarray([cutted_I, cutted_Q]).T
        toSaveObj.append(cutted_IQ)
    # pickle.dump(toSaveObj, open(dest_path, 'wb'))
    plt.figure()
    for index,obj in enumerate(toSaveObj):
        if index >= 8:
            break
        obj = obj - np.roll(obj, 1)
        obj = obj[1:]
        obj = norm(obj, ord=2, axis=1)
        obj = obj - np.roll(obj, 1)
        obj = obj[1:]
        obj = np.abs(obj)
        # data = normalize(data)
        plt.subplot(8,1,index+1)
        plt.plot(obj)
    plt.show()

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


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def test():
    cut('../server/2018-12-12-20-00-31/temp/handwriting/dingfeng/1.pcm',
        '../server/2018-12-12-20-00-31/temp/handwriting/dingfeng/1.pcm')


if __name__ == '__main__':
    main()
    # test()
