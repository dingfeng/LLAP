# -*- coding: UTF-8 -*-
# filename: cut_data date: 2018/12/11 15:13  
# author: FD 
import numpy as np
from scipy.signal import butter, lfilter, find_peaks_cwt
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.seasonal import seasonal_decompose
fs = 48000
freq = 20000


def main():
    source_dir='../dataset/data20-10-mimic/raw/dingfeng-dengyufeng'
    dest_dir='../dataset/data20-10-mimic/test/dingfeng-dengyufeng'
    cut_dir(source_dir, dest_dir)



def cut_dir(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith('pcm'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, ''.join([filename[:-4],'.pkl']))
            cut(source_path,dest_path)


def cut(source_path, dest_path):
    data = np.memmap(source_path, dtype=np.float32, mode='r')
    cutted_Is=[]
    cutted_Qs=[]
    # for i in range(1,4):
    #     biase=np.pi/2 * i
    I = getI(data, freq,0)
    I = move_average(I)
    decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
    # decompositionI.plot()
    plt.show()
    I = decompositionI.trend
    fig=plt.figure()
    plt.title(source_path)
    plt.plot(I)
    points = plt.ginput(5, timeout=0)
    points_len=len(points)
    print('point length {}'.format(points_len))
    if(points_len == 0):
        return
    cutted_I = I[int(points[0][0]):int(points[1][0])]
    cutted_Is.append(cutted_I)
    plt.close(fig)
    for i in range(1,16):
        I = getI(data, freq, i*np.pi/8)
        I = move_average(I)
        decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
        I=decompositionI.trend
        cutted_I = I[int(points[0][0]):int(points[1][0])]
        cutted_Is.append(cutted_I)
    pickle.dump({'I': cutted_Is}, open(dest_path, 'wb'))


def getI(data, f,biase):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times+biase) * data
    return mulCos


def getQ(data, f,biase):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times+biase) * data
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

# def removeTrend(data):


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def test():
    cut('../server/2018-12-12-20-00-31/temp/handwriting/dingfeng/1.pcm','../server/2018-12-12-20-00-31/temp/handwriting/dingfeng/1.pcm')


if __name__ == '__main__':
    main()
    # test()


