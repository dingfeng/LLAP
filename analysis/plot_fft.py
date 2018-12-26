# -*- coding: UTF-8 -*-
# filename: plot_fft date: 2018/11/24 19:15  
# author: FD
import scipy.fftpack as fftp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


def filter(data):
    return butter_bandpass_filter(data, 17000, 23800, 48000, order=3)


def main():
    filepath = '../server/2018-12-25-20-37-29/temp/dingfeng/test/1.pcm'
    data = np.memmap(filepath, dtype=np.float32, mode='r')
    data = filter(data)
    xf = np.arange(len(data)) / len(data) * 48000
    yf = fftp.fft(data, len(data))
    yf = np.abs(yf)
    baseF = 17350
    deltaF = 700
    fn = 8
    fslist = []
    currentF = baseF
    fslist.append(currentF)
    for i in range(fn - 1):
        currentF += deltaF
        fslist.append(currentF)
    for index, fc in enumerate(fslist):
        print(index)
        for value_index, value in enumerate(xf):
            if value == fc:
                print('index {} fc {} freq {} fft_energy {}'.format(index, fc, xf[value_index], yf[value_index]))
    # for i in range(20):
    #     yf[np.argmax(yf)] = 0
    plt.plot(xf, yf)
    plt.show()
    pass


if __name__ == '__main__':
    main()
