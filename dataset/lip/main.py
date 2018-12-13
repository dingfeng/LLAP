# -*- coding: UTF-8 -*-
# filename: main date: 2018/12/12 20:34  
# author: FD 
import numpy as np
import matplotlib.pyplot as plt

fs = 48000
freq = 20000


def main():
    for i in range(1,10):
        save_fig(i)
    pass


def save_fig(count):
    source_path = 'zhangqian/' + str(count) + ".pcm"
    data=np.memmap(source_path,dtype=np.float32,mode='r')
    I = getI(data, freq)
    I = move_average(I)
    Q = getQ(data, freq)
    Q = move_average(Q)
    plt.figure()
    plt.title(str(count))
    plt.subplot(211)
    plt.plot(I)
    plt.subplot(212)
    plt.plot(Q)
    plt.savefig(str(count)+".png")

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


if __name__ == '__main__':
    main()
