# -*- coding: UTF-8 -*-
# filename: plot_fft date: 2018/11/23 20:09  
# author: FD 
import numpy as np
import pylab
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks

fs = 48000


def main():
    getData()
    return


def getData():
    filepath = 'F:/rfid实验室/新研究点/超声波/论文实现/llap/server/2019-05-04-11-24-21/temp/zhangqian/g/1.pcm'
    data = np.memmap(filepath, dtype=np.float32, mode='r')
    # data = butter_bandpass_filter(data, 18000, 22000, fs)
    # data = data[130000:]
    fc=17350+700*0
    data=butter_bandpass_filter(data,fc-300,fc+300,48000)
    f = fc
    # downI = move_average()
    I = getI(data, f)
    I = move_average(I)
    # I=move_average(I)
    # I=move_average(I)
    Q = getQ(data, f)
    # Q = move_average(Q)
    # Q=move_average(Q)
    # butter_bandpass_filter()
    start_point = 0
    I = I[start_point:]
    Q = move_average(Q)
    Q = Q[start_point:]
    # data=Q
    # Q=butter_lowpass_filter(Q,200,fs)
    # data = Q
    # xf = np.arange(len(data)) / len(data) * 48000
    # yf = fftp.fft(data, len(data))
    # yf = np.abs(yf)
    # for i in range(20):
    #     yf[np.argmax(yf)] = 0
    # plt.plot(xf, yf)
    # plt.show()
    # # plt.subplot(311)
    # plt.plot(I)
    # plt.subplot(312)
    # tmp=Q[220000:260000]
    # peaks=find_peaks(tmp)[0]
    # print(peaks)
    # plt.figure()
    # plt.plot(tmp)
    # plt.scatter(peaks,tmp[peaks])
    # plt.figure()
    # diff=np.roll(peaks,-1)-peaks
    # diff=diff[:10]
    # print(np.mean(diff))
    # plt.plot(diff)
    # plt.show()
    # plt.plot(Q)
    # plt.subplot(313)
    # plt.plot(I, Q)
    # decomposition = seasonal_decompose(data, freq=3006, two_sided=False)
    # decomposition.plot()
    # plt.show()
    # getPhase1(I,Q)
    # plt.plot(I)
    # plt.show()
    # downI = downI[35 + 30:][0:200]
    # downI = removeDC(downI)
    # downQ = move_average(getQ(data, f))
    # downQ = downQ[35 + 30:][0:200]
    # downQ = removeDC(downQ)
    # print(" I max {} Q max {}".format(np.max(downI), np.max(downQ)))
    # phases = np.zeros(len(downI))
    # for i in np.arange(len(phases)):
    #     phases[i] = getPhase(downQ[i], downI[i])
    # phases = np.unwrap(phases)
    # distances=distanceLine(phases,f)
    # plt.plot(distances)
    # plt.plot(downQ,downI)
    decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
    trendQ = decompositionQ.trend
    trendQ=trendQ[10:]
    decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
    trendI = decompositionI.trend
    trendI=trendI[10:]
    plt.figure()
    plt.subplot(211)
    plt.plot(trendQ)
    plt.subplot(212)
    plt.plot(trendI)
    # phase = getPhase1(trendI, trendQ)
    # plt.plot(phase)
    plt.show()
    return data


def getPhase1(I, Q):
    derivativeQ = getDerivative(Q)
    derivativeI = getDerivative(I)
    # phase=np.unwrap(2*())+np.pi/2))/2
    # distance=distanceLine(phase,20000)
    # plt.plot(distance)
    # plt.show()
    derivativeQ = np.asarray(derivativeQ)
    derivativeQ[np.where(derivativeQ==0)]=0.000001
    arcValue = np.arctan(-np.asarray(derivativeI) / (derivativeQ))
    newData = unwrap(arcValue)
    plt.plot(newData)
    plt.show()


def unwrap(data):
    resultData = []
    diffs = np.roll(data, -1) - data
    diffs = diffs[:len(data) - 1]
    first_value = data[0]
    resultData.append(first_value)
    previous_value = first_value
    current_value=None
    for diff in diffs:
        if diff > np.pi / 2:
            current_value = previous_value + diff - np.pi
            resultData.append(current_value)
        elif diff < -np.pi / 2:
            current_value = previous_value + diff + np.pi
            resultData.append(current_value)
        else:
            current_value=previous_value+diff
            resultData.append(current_value)
        previous_value = current_value
    return np.asarray(resultData)

def getDerivative(data):
    derivativeQ = []
    for i in range(len(data) - 1):
        derivativeQ.append((data[i + 1] - data[i]))
    return derivativeQ


def removeDC(data):
    return data - np.mean(data)


def distanceLine(phase, freq):
    distances = np.zeros(len(phase) - 1)
    for i in np.arange(1, len(phase)):
        phaseDiff = phase[0] - phase[i]
        distanceDiff = 343 / (2 * np.pi * freq) * phaseDiff
        distances[i - 1] = distanceDiff
    distances = distances / 2
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
    win_size = 300
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result


def getI(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times) * data
    return mulCos


def getQ(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times) * data
    return mulSin


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


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


if __name__ == '__main__':
    main()
