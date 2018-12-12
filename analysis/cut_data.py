# -*- coding: UTF-8 -*-
# filename: cut_data date: 2018/12/11 15:13  
# author: FD 
import numpy as np
from scipy.signal import butter, lfilter, find_peaks_cwt
import matplotlib.pyplot as plt
import pickle
import os

fs = 48000
freq = 20000


def main():
    source_dir='../dataset/raw/dingfeng2'
    dest_dir='../dataset/cutted/dingfeng2'
    cut_dir(source_dir, dest_dir)



def cut_dir(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith('pcm'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, ''.join([filename[:-4],'.pkl']))
            cut(source_path,dest_path)


def cut(source_path, dest_path):
    data = np.memmap(source_path, dtype=np.float32, mode='r')
    I = getI(data, freq)
    I = move_average(I)
    Q = getQ(data, freq)
    Q = move_average(Q)
    fig=plt.figure()
    plt.title(source_path)
    plt.subplot(211)
    plt.plot(I)
    plt.subplot(212)
    plt.plot(Q)
    points = plt.ginput(5, timeout=0)
    cutted_I = I[int(points[0][0]):int(points[1][0])]
    cutted_Q = Q[int(points[0][0]):int(points[1][0])]
    plt.close(fig)
    pickle.dump({'I': cutted_I, 'Q': cutted_Q}, open(dest_path, 'wb'))


def getI(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times) * data
    return mulCos


def getQ(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times) * data
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


if __name__ == '__main__':
    main()
