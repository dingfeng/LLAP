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


fs = 48000
freq = 20000
NUM_CLUSTERS = 5
freqs = [17350 + i * 700 for i in range(8)]
I_multipliers = None
Q_multipliers = None
filter_half_width = 100  # +-100Hz

def main():
    init_IQ_multipliers()
    dataset_dir = '../dataset2'
    names = os.listdir(dataset_dir)
    for name in names:
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
    sub_dirs = ['music', 'reference', 'walk']
    to_dump = {}
    dest_filepath = os.path.join(dir_path, 'feature.pkl')
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(dir_path, sub_dir)
        filenames = os.listdir(sub_dir_path)
        features = []
        for filename in filenames:
            filepath = os.path.join(sub_dir_path, filename)
            feature = extract_feature(filepath)
            features.append(feature)
        to_dump[sub_dir] = features
    pickle.dump(to_dump, open(dest_filepath, 'wb'))


def extract_feature(source_path):
    try:
        data = np.load(open(source_path, 'rb'))
    except:
        print('error file {}'.format(source_path))
        return
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
