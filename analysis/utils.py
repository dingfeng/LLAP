# -*- coding: UTF-8 -*-
# filename: utils date: 2018/12/20 14:24  
# author: FD 
import scipy.fftpack as fftp
import numpy as np
import matplotlib.pyplot as plt

def plot_fft(data,fs=48000):
    xf = np.arange(len(data)) / len(data) * fs
    yf = fftp.fft(data, len(data))
    yf = np.abs(yf)
    # for i in range(20):
    #     yf[np.argmax(yf)] = 0
    plt.plot(xf, yf)
    plt.show()