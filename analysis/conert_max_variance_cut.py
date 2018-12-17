# -*- coding: UTF-8 -*-
# filename: conert_max_variance_cut date: 2018/12/17 18:41  
# author: FD 
import numpy as np
import pickle
import os


def main():
    source_dir_path = '../dataset/data20-10/cutted'
    dest_dir_path = '../dataset/data20-10/max_variance_cutted'
    convert_dir(source_dir_path, dest_dir_path)
    pass


def convert_dir(dir_path, dest_dir_path):
    label_names = os.listdir(dir_path)
    for label_name in label_names:
        label_path = os.path.join(dir_path, label_name)
        dest_label_path = os.path.join(dest_dir_path, label_name)
        if not os.path.isdir(dest_label_path):
            os.makedirs(dest_label_path)
        for filename in os.listdir(label_path):
            filepath = os.path.join(label_path, filename)
            dest_filepath = os.path.join(dest_label_path, filename)
            convert(filepath, dest_filepath)


def convert(source, dest):
    onedata = np.load(open(source, 'rb'))['I']
    max_variance = -1
    max_index = -1
    for i in range(len(onedata)):
        variance = np.var(onedata[i])
        if (variance > max_variance):
            max_variance = variance
            max_index = i
    final_data = onedata[max_index]
    pickle.dump([final_data, -final_data], open(dest, 'wb'))


if __name__ == '__main__':
    main()
