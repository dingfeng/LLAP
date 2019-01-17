# -*- coding: UTF-8 -*-
# filename: cut_data date: 2018/12/26 19:19  
# author: FD 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import norm
import pandas
from scipy.stats import pearsonr
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import pickle
import os
from scipy.signal import butter, lfilter
import time
from analysis.utils import plot_fft
import matplotlib.pyplot as plt

fs = 48000
freq = 20000
NUM_CLUSTERS = 5
freqs = [17350 + i * 700 for i in range(8)]
I_multipliers = None
Q_multipliers = None
filter_half_width = 100  # +-100Hz


def init_IQ_multipliers():
    global I_multipliers
    global Q_multipliers
    max_len = 48000 * 15
    times = np.arange(0, max_len) * 1 / fs
    line_num = 64
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


names = [#'anna', 'dengyufeng', 'dingfeng', 'huangsi',
         #'jianghao', 'qingpeijie', 'xuhuatao',
         'yinjunhao', 'yuyinggang',
         #'zhangqian', 'zhaorun', 'zhuyan'
         ]


def main():
    init_IQ_multipliers()
    # cut_dir('../dataset/handwriting-lab-1/raw/chenbo', '../dataset/handwriting-lab-1/cutted/chenbo-forged')
    # cut_dir('../dataset/handwriting-lab-1/raw/chenbo', '../dataset/handwriting-lab-1/cutted/test')
    for i in range(len(names)):
        name = names[i]
        print('name : {}'.format(name))
        mimic_cut('../dataset/handwriting-lab-1/mimic-raw/' + name,
                '../dataset/handwriting-lab-1/mimic_cutted_arranged3/' + name)
        # cut_dir('../dataset/handwriting-lab-1/raw/'+name,'../dataset/handwriting-lab-1/cutted2/'+name)
    # cut_dir('../dataset/test/raw/dingfeng','../dataset/test/cutted/dingfeng-forged-forged')
    # cut_undergraduate()
    return


def cut_undergraduate():
    root_dir = '../dataset/handwriting-lab-3/'
    dirnames = os.listdir(root_dir + '/raw')
    for dirname in dirnames:
        mimic_cut(os.path.join(root_dir, 'raw', dirname), os.path.join(root_dir, 'cutted', dirname))


def mimic_cut(source_dir, dest_dir):
    mimic_dirs = os.listdir(source_dir)
    for mimic_dir in mimic_dirs:
        source_mimic_dir_path = os.path.join(source_dir, mimic_dir)
        dest_mimic_dir_path = os.path.join(dest_dir, mimic_dir)
        if not os.path.isdir(dest_mimic_dir_path):
            os.makedirs(dest_mimic_dir_path)
        cut_dir(source_mimic_dir_path, dest_mimic_dir_path)


def cut_dir(source_dir, dest_dir):
    count = 0
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('pcm'):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, ''.join([filename[:-4], '.pkl']))
            cut_file(source_path, dest_path)
            count += 1
            print('count {}'.format(count))


def cut_file(source_filepath, dest_filepath):
    global cluster_count
    global freqs
    global fs
    data = np.memmap(source_filepath, dtype=np.float32, mode='r')
    data = data[54000:]
    previous = None
    previous_added = False
    previous_time = time.time()
    final_features = []
    total_average_time = 0.0
    total_decompose_time = 0.0
    total_diff_time = 0.0
    total_correlation_time = 0.0
    for freq_index, freq in enumerate(freqs):
        freq_data = data  # butter_lowpass_filter(data,100,fs)#butter_bandpass_filter(data, freq - filter_half_width, freq + filter_half_width, fs)
        cutted_IQs = []
        velocities = []
        # repeat_time = time.time()
        for i in range(64):
            I = getI(freq_data, freq_index, i)
            I = butter_lowpass_filter(I, 100, fs)[200:][::300]
            decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
            I = decompositionI.trend[11:]
            Q = getQ(freq_data, freq_index, i)
            Q = butter_lowpass_filter(Q, 100, fs)[200:][::300]
            decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
            Q = decompositionQ.trend[11:]
            diff_time = time.time()
            IQ = np.asarray([I, Q]).T
            IQ0 = IQ - np.roll(IQ, 1, axis=0)
            IQ1 = (IQ - np.roll(IQ, 2, axis=0)) / 2
            IQ = (IQ0 + IQ1) / 2
            IQ = IQ[4:-4, :]
            IQ[:, 0] = IQ[:, 0] - IQ[0, 0]
            IQ[:, 1] = IQ[:, 1] - IQ[0, 1]
            IQ = norm(IQ, ord=2, axis=1)
            velocity = IQ
            IQ0 = IQ - np.roll(IQ, 1, axis=0)
            IQ1 = (IQ - np.roll(IQ, 2, axis=0)) / 2
            IQ = (IQ0 + IQ1) / 2
            IQ = IQ[4:-4]
            cutted_IQ = IQ
            total_diff_time += time.time() - diff_time
            correlation_time = time.time()
            if previous is not None:
                correlation = get_correlation(cutted_IQ, previous)
                # print(correlation)
                if correlation > 0.95 and previous is not None:
                    cutted_IQs.append(cutted_IQ)
                    velocities.append(velocity)
                    if not previous_added:
                        cutted_IQs.append(previous)
                        velocities.append(velocity)
                        previous_added = True
                else:
                    previous_added = False
            total_correlation_time += time.time() - correlation_time
            previous = cutted_IQ
        vars = []
        for cutted_IQ in cutted_IQs:
            vars.append(np.var(cutted_IQ))
        index = np.argsort(vars)[len(vars) - 1]
        final_features.append(cutted_IQs[index])
        final_features.append(velocities[index])
    print('total time {}'.format(time.time() - previous_time))
    print('average {} decompose {} diff {} correlation {}'.format(total_average_time, total_decompose_time,
                                                                  total_diff_time, total_correlation_time))
    pickle.dump(final_features, open(dest_filepath, 'wb'))


def get_PAM_distance(index0, index1):
    global distances_mat
    print('index 0 {} index 1 {}'.format(index0, index1))
    return distances_mat[index0, index1]


def getI(data, freq_index, bias_index):
    return I_multipliers[freq_index][bias_index][:len(data)] * data


def getQ(data, freq_index, bias_index):
    return Q_multipliers[freq_index][bias_index][:len(data)] * data


def get_correlation(data0, data1):
    short_data = data0
    long_data = data1
    if len(long_data) < len(short_data):
        temp = short_data
        short_data = long_data
        long_data = temp
    lags = [i for i in range(len(long_data) - len(short_data) + 1)]
    max_pearson = -2
    short_data_len = len(short_data)
    for lag in lags:
        pearson_value = pearsonr(long_data[lag:lag + short_data_len], short_data)[0]
        max_pearson = max(max_pearson, pearson_value)
    return max_pearson


def move_average(data):
    win_size = 300
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    reshape_time = time.time()
    data = data.reshape((new_len, win_size))
    # print('reshape time {}'.format(time.time()-reshape_time))
    inner_mean_time = time.time()
    mean_result = np.mean(data, axis=1)
    # print('inner mean time {}'.format(time.time()-inner_mean_time))
    return mean_result


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
