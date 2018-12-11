# -*- coding: UTF-8 -*-
# filename: liphase date: 2018/11/27 11:06  
# author: FD 
# -*- coding: UTF-8 -*-
# filename: plot_fft date: 2018/11/23 20:09
# author: FD
import numpy as np
import pylab
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
from analysis.CIC import *
from scipy.signal import butter, lfilter, find_peaks_cwt

freq = 48000


def main():
    getData()
    return


def getData():
    filepath = '../server/2/temp/distance/dingfeng/dingfeng.pcm'
    data = np.memmap(filepath, dtype='h', mode='r')
    f=17000 + 175
    downI = move_average(getI(data, f))
    # downI = downI[50:200]
    # downI = removeDC(downI)
    # downQ = move_average(getQ(data, f))
    # downQ = downQ[50:200]
    # downQ = removeDC(downQ)
    # print(" I max {} Q max {}".format(np.max(downI), np.max(downQ)))
    # phases = np.zeros(len(downI))
    # for i in np.arange(len(phases)):
    #     phases[i] = getPhase(downQ[i], downI[i])
    # phases = np.unwrap(phases)
    # distances=distanceLine(phases,f)
    # plt.plot(distances)
    plt.plot(downI)
    plt.show()
    return data


def removeDC(data):
    return data - np.mean(data)


def distanceLine(phase, freq):
    distances=np.zeros(len(phase)-1)
    for i in np.arange(1, len(phase)):
        phaseDiff = phase[0] - phase[i]
        distanceDiff = 343 / (2 * np.pi * freq) * phaseDiff
        distances[i-1]=distanceDiff
    return distances

def getPhase(Q, I):
    if I == 0 and Q > 0:
        return np.pi / 2
    elif I == 0 and Q < 0:
        return 3 / 2 * np.pi
    elif Q == 0 and I > 0:
        return 0
    elif Q == 0 and I < 0:
        return np.pi
    tanValue = Q / I
    tanPhase = np.arctan(tanValue)
    resultPhase = 0
    if I > 0 and Q > 0:
        resultPhase = tanPhase
    elif I < 0 and Q > 0:
        resultPhase = np.pi + tanPhase
    elif I < 0 and Q < 0:
        resultPhase = np.pi + tanPhase
    elif I > 0 and Q < 0:
        resultPhase = 2 * np.pi + tanPhase
    return resultPhase


def move_average(data):
    win_size = 1300
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result


def getI(data, f):
    times = np.arange(0, len(data)) * 1 / freq
    mulCos = np.cos(2 * np.pi * f * times) * data
    return mulCos


def getQ(data, f):
    times = np.arange(0, len(data)) * 1 / freq
    mulSin = -np.sin(2 * np.pi * f * times) * data
    return mulSin


if __name__ == '__main__':
    main()
