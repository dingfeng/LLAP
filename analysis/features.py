# -*- coding: UTF-8 -*-
# filename: features date: 2018/12/18 22:48  
# author: FD 
import numpy as np
from scipy.linalg import norm
from scipy.signal import find_peaks

def get_feature(data):
    # 长度，最大值，最小值，最大值-最小值，总和，平均值，标准差，变异系数，偏度，峰度，极差
    length = len(data)
    max_value = np.max(data)
    min_value = np.min(data)
    margin = max_value - min_value
    sum = np.sum(data)
    average = sum / length
    std = np.std(data)
    cv = std / average
    skewness = np.mean((data - average) ** 3) / np.power(np.mean((data - average) ** 2), 1.5)
    energy=norm(data,ord=2)
    # peak_count=len(find_peaks(data)[0])
    # print(peak_count)
    return np.asarray([length, max_value, margin, average, std, cv, skewness,energy])
