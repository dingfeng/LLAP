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
import time

records = {}


def main():
    global records
    source_dir = '../dataset/handwriting-lab-1/cutted-chord'
    dest_dir = '../dataset/handwriting-lab-1/feature-chord'
    cut_to_dir(source_dir, dest_dir)
    pickle.dump(records,open('feature_time_records','wb'))
    pass


def test():
    source_path = '../dataset/handwriting-lab-1/cutted-chord/dingfeng/2.pkl'
    dest_path = None
    cut_feature_to_file(source_path, dest_path)


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
    global records
    key = source_path.split('/')[-2] + '-' + source_path.split('/')[-1][:-4]
    try:
        data = np.load(open(source_path, 'rb'))
    except:
        print('error file {}'.format(source_path))
        return
    final_dct_result = None
    start_time = time.time()
    for index, item in enumerate(data):
        dct_result = dct(item)
        # plt.figure(figsize=(10,7))
        # print('a figure')
        # # plt.figure(figsize=(10,5))
        # plt.subplot(211)
        # plt.plot(item*100000)
        # plt.ylabel('Velocity (1e-5)',fontdict={'style': 'normal', 'weight': 'bold','size':20})
        # plt.xticks(fontsize=17,fontname='normal')
        # plt.yticks(fontsize=17,fontname='normal')
        # plt.legend(prop={'size': 20})
        if len(dct_result) < 200:
            dct_result = np.pad(dct_result, (0, -len(dct_result) + 200), 'constant', constant_values=0)
        dct_result = dct_result.reshape(-1, 1)
        ss = StandardScaler()
        ss.fit(dct_result)
        # plt.subplot(212)
        # plt.plot(ss.transform(dct_result),label='dct')
        # plt.xlabel('Data Index', fontdict={'style': 'normal', 'weight': 'bold', 'size': 20})
        # plt.ylabel('DCT (normalized)',fontdict={'style': 'normal', 'weight': 'bold','size':20})
        # plt.xticks(fontsize=17, fontname='normal')
        # plt.yticks(fontsize=17, fontname='normal')
        # # plt.legend(prop={'size': 20})
        # plt.tight_layout()
        # plt.savefig('velocity-dct.pdf', dpi=100)
        # plt.show()
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
    end_time = time.time()
    pickle.dump(final_dct_result, open(dest_path, 'wb'))
    duration = end_time - start_time
    records[key] = duration


if __name__ == '__main__':
    main()
    # test()
