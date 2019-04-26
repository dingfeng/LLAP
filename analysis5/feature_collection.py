# -*- coding: UTF-8 -*-
# filename: feature_collection date: 2019/1/16 11:40  
# author: FD 
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from scipy.linalg import norm


def main():
    # source_dir = '../dataset/handwriting-lab-3/mimic-cutted-chord-arranged'
    # dest_dir = '../dataset/handwriting-lab-3/forged-feature-chord-40'
    source_dir = '../dataset/handwriting-lab-3/cutted-chord'
    dest_dir = '../dataset/handwriting-lab-3/feature-chord-40'
    cut_to_dir(source_dir, dest_dir)
    pass


def cut_to_dir(source_dir, dest_dir):
    dir_names = os.listdir(source_dir)
    for dir_name in dir_names:
        dir_path = os.path.join(source_dir, dir_name)
        filenames = os.listdir(dir_path)
        dest_dir_path = os.path.join(dest_dir, dir_name)
        if not os.path.isdir(dest_dir_path):
            os.makedirs(dest_dir_path)
        for filename in filenames:
            if filename.startswith('index'):
                continue
            filepath = os.path.join(dir_path, filename)
            dest_filepath = os.path.join(dest_dir_path, filename)
            cut_feature_to_file(filepath, dest_filepath)


def cut_feature_to_file(source_path, dest_path):
    try:
        data = np.load(open(source_path, 'rb'),allow_pickle=True)
    except:
        print('error file {}'.format(source_path))
        return
    final_dct_result = None
    for index, item in enumerate(data):
        # item=item[:min(len(data),value_dict[dir_name])]
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
    pickle.dump(final_dct_result, open(dest_path, 'wb'))
    return len(data[0][:])


if __name__ == '__main__':
    main()
