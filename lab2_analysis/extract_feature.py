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
    dataset_dir = '../replay-attack-dataset'
    names = os.listdir(dataset_dir)
    for name in names:
        if  name == 'dengyufeng':
            continue
        dir_path = os.path.join(dataset_dir, name)
        cut_dir(dir_path)



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


def cut_dir(dir_path):
    # sub_dirs = ['music', 'reference', 'walk', '0day','1day','7day']
    # sub_dirs = ['distance0', 'distance1', 'distance2']
    sub_dirs = ['replayattack', 'replayattack-1']
    to_dump = {}
    dest_filepath = os.path.join(dir_path, 'feature.pkl')
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(dir_path, sub_dir)
        filenames = os.listdir(sub_dir_path)
        features = []
        for filename in filenames:
            filepath = os.path.join(sub_dir_path, filename)
            feature = cut_file(filepath)
            features.append(feature)
        to_dump[sub_dir] = features
    pickle.dump(to_dump, open(dest_filepath, 'wb'))


def extract_feature(data):
    final_dct_result = None
    for index, item in enumerate(data):
        item = (item - np.min(item)) / (np.max(item) - np.min(item))
        dct_result = dct(item)
        if len(dct_result) < 200:
            dct_result = np.pad(dct_result, (0, -len(dct_result) + 200), 'constant', constant_values=0)
        dct_result = dct_result.reshape(-1, 1)
        dct_result = dct_result[:200, :]
        if index % 2 == 1:
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
            # I=I[10:]
            # trend=decompositionI.trend[10:]
            # seasonal=decompositionI.seasonal[10:]
            # resid=decompositionI.resid[10:]
            # plt.figure(figsize=(10, 6))
            # plt.plot(x, results[:, 0], lw=2, marker='o', c='r', markersize=12, label='AUC')
            # plt.xlabel('CNN Filter Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
            # plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
            # plt.xticks(fontsize=20, fontname='normal')
            # plt.yticks(fontsize=20, fontname='normal')
            # plt.legend(prop={'size': 22}, loc='center right')
            # plt.tight_layout()
            # plt.savefig('./filter_number_auc_eer.pdf')
            # plt.show()
            # plt.figure(figsize=(15,20))
            # x=np.arange(len(I))/160
            # ax1=plt.subplot(411)
            # plt.plot(x,I,lw=2)
            # plt.setp(ax1.get_xticklabels(), visible=False)
            # plt.ylabel('Original Sequence', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
            # plt.yticks(fontsize=20, fontname='normal')
            # ax2=plt.subplot(412)
            # plt.plot(x,trend,lw=2)
            # plt.setp(ax2.get_xticklabels(), visible=False)
            # plt.ylabel('Trend Sequence', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
            # plt.yticks(fontsize=20, fontname='normal')
            # ax3=plt.subplot(413)
            # plt.plot(x,seasonal,lw=2)
            # plt.setp(ax3.get_xticklabels(), visible=False)
            # plt.ylabel('Seasonal Sequence', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
            # plt.yticks(fontsize=20, fontname='normal')
            # plt.subplot(414)
            # plt.plot(x,resid,lw=2)
            # plt.xlabel('Time (Seconds)', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
            # plt.xticks(fontsize=20, fontname='normal')
            # plt.yticks(fontsize=20, fontname='normal')
            # plt.ylabel('Residual Sequence', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
            # plt.tight_layout()
            # plt.savefig('./std-effect.pdf')
            # plt.show()


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
    feature = extract_feature(preprocess_results)
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