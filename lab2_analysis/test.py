# -*- coding: UTF-8 -*-
# filename: extract_feature date: 2019/4/16 10:20
# author: FD
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import pickle
from scipy.fftpack import dct
import os
from scipy.signal import butter, lfilter
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import norm
import matplotlib.pyplot as plt

fs = 48000
freq = 20000
freqs = [17350 + i * 700 for i in range(8)]
I_multipliers = None
Q_multipliers = None
filter_half_width = 100  # +-100Hz


def main():
    init_IQ_multipliers()
    data=cut_file('1.pcm')
    data1=np.load('1.pkl','rb')
    return


def init_IQ_multipliers():
    global I_multipliers
    global Q_multipliers
    max_len = 48000 * 15
    times = np.arange(0, max_len) * 1 / fs
    line_num = 1
    I_multipliers = np.zeros((len(freqs), line_num, max_len))
    Q_multipliers = np.zeros((len(freqs), line_num, max_len))
    for freq_index, freq in enumerate(freqs):
        # generate I multipliers
        for i in range(line_num):
            bias = i * np.pi / 32
            mul_cos = np.cos(2 * np.pi * freq * times + bias)
            I_multipliers[freq_index, i, :] = mul_cos
        # generate Q multipliers
        for i in range(line_num):
            bias = i * np.pi / 32
            mul_sin = -np.sin(2 * np.pi * freq * times + bias)
            Q_multipliers[freq_index, i, :] = mul_sin



def extract_feature(data):
    final_dct_result = None
    for index, item in enumerate(data):
        dct_result = dct(item)
        if len(dct_result) < 200:
            dct_result = np.pad(dct_result, (0, -len(dct_result) + 200), 'constant', constant_values=0)
        dct_result = dct_result.reshape(-1, 1)
        ss = StandardScaler()
        ss.fit(dct_result)
        dct_result = ss.transform(dct_result)[:200, :]
        if index % 2 == 0:
            dct_result[40:200, :] = 0
        try:
            if final_dct_result is None:
                final_dct_result = dct_result
            else:
                final_dct_result = np.hstack((final_dct_result, dct_result))
        except:
            print('error')
    return final_dct_result


def cut_file(source_filepath):
    global freqs
    global fs
    data = np.memmap(source_filepath, dtype=np.float32, mode='r')
    data = data[54000:]
    preprocess_results = []
    for freq_index, freq in enumerate(freqs):
        freq_data = data  # butter_lowpass_filter(data,100,fs)#butter_bandpass_filter(data, freq - filter_half_width, freq + filter_half_width, fs)
        for i in range(1):
            I = getI(freq_data, freq_index, i)
            I = butter_lowpass_filter(I, 100, fs)[200:][::300]
            decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
            I = decompositionI.trend[10:]
            Q = getQ(freq_data, freq_index, i)
            Q = butter_lowpass_filter(Q, 100, fs)[200:][::300]
            decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
            Q = decompositionQ.trend[10:]
            IQ = np.asarray([I, Q]).T
            IQ = IQ - np.roll(IQ, 1, axis=0)
            IQ = IQ[1:, :]
            IQ = norm(IQ, ord=2, axis=1)
            velocity = IQ
            IQ0 = IQ - np.roll(IQ, 1, axis=0)
            IQ1 = (IQ - np.roll(IQ, 2, axis=0)) / 2
            IQ = (IQ0 + IQ1) / 2
            IQ = IQ[4:-4]
            cutted_IQ = IQ
            preprocess_results.append(cutted_IQ)
            preprocess_results.append(velocity)
    feature = preprocess_results
    return feature


def getI(data, freq_index, bias_index):
    return I_multipliers[freq_index][bias_index][:len(data)] * data


def getQ(data, freq_index, bias_index):
    return Q_multipliers[freq_index][bias_index][:len(data)] * data


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
