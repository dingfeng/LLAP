# -*- coding: UTF-8 -*-
# filename: generate_dataset date: 2019/1/16 13:30  
# author: FD 
import numpy as np
import os
import pickle
import time

data_set = []
label_set = []
deep_label_set=[]
names = None
records={}

def main():
    global names
    global records
    names = ['anna', 'chenbo',
             'chenhao',
             'dengyufeng', 'dingfeng', 'huangsi', 'jianghao', 'qingpeijie', 'xuhuatao', 'yinjunhao', 'yuyinggang',
             'zhangqian', 'zhaorun', 'zhuyan']
    for name in names:
        generate_by_name(name,21)
    indexes = np.arange(len(data_set))
    np.random.shuffle(indexes)
    test_rate = 0.3
    test_count = int(len(indexes) * test_rate)
    train_data_set = []
    train_label_set = []
    test_data_set = []
    test_label_set = []
    for i in range(test_count):
        test_data_set.append(data_set[indexes[i]])
        test_label_set.append(label_set[indexes[i]])
    for i in range(test_count, len(indexes)):
        train_data_set.append(data_set[indexes[i]])
        train_label_set.append(label_set[indexes[i]])
    pickle.dump({'train_data_set': train_data_set, 'train_label_set': train_label_set, 'test_data_set': test_data_set,
                 'test_label_set': test_label_set}, open('dataset-1-tr.pkl', 'wb'))
    pickle.dump(records,open('generate_dataset_records','wb'))

def generate_dataset():
    global names
    global data_set
    global label_set
    global deep_label_set
    for reference_amount in range(21,22):
        for k in range(20):
            names = None
            data_set = []
            label_set = []
            deep_label_set = []
            names = ['anna', 'chenbo',
                     'chenhao',
                     'dengyufeng', 'dingfeng', 'huangsi', 'jianghao', 'qingpeijie', 'xuhuatao', 'yinjunhao', 'yuyinggang',
                     'zhangqian', 'zhaorun', 'zhuyan']
            for name in names:
                generate_by_name(name,reference_amount)
            indexes = np.arange(len(data_set))
            np.random.shuffle(indexes)
            test_rate = 0.3
            test_count = int(len(indexes) * test_rate)
            train_data_set = []
            train_label_set = []
            test_data_set = []
            test_label_set = []
            deep_test_label_set=[]
            for i in range(test_count):
                test_data_set.append(data_set[indexes[i]])
                test_label_set.append(label_set[indexes[i]])
                deep_test_label_set.append(deep_label_set[indexes[i]])
            for i in range(test_count, len(indexes)):
                train_data_set.append(data_set[indexes[i]])
                train_label_set.append(label_set[indexes[i]])
            dir_path='O:/evaluation/test-dataset/{}'.format(reference_amount)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            pickle.dump(
                {'train_data_set': train_data_set, 'train_label_set': train_label_set, 'test_data_set': test_data_set,
                 'test_label_set': test_label_set,'deep_test_label_set':deep_test_label_set}, open('O:/evaluation/test-dataset/{}/dataset-{}.pkl'.format(reference_amount,k + 1), 'wb'))

def generate_by_name(name,template_count):
    global names
    global data_set
    global deep_label_set
    global records
    dir_path = '../dataset/handwriting-lab-1/feature-chord/' + name
    filenames = os.listdir(dir_path)
    templates = []
    for i in range(template_count):
        data = np.load(open(dir_path + '/' + filenames[i], 'rb'))
        templates.append(data)
    min_data = np.zeros(data.shape)
    max_data = np.zeros(data.shape)
    mean_data = np.zeros(data.shape)
    sum_data = np.zeros(data.shape)
    local_dataset = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            total = 0
            min_value = 100000
            max_value = -100000
            for template in templates:
                total += template[i][j]
                min_value = min(min_value, template[i][j])
                max_value = max(max_value, template[i][j])
            min_data[i][j] = min_value
            max_data[i][j] = max_value
            sum_data[i][j] = total
            mean_data[i][j] = total / len(templates)
    max_record = np.zeros(6)
    for i in range(len(max_record)):
        max_record[i] = -10000
    for i in range(min(90,len(filenames)-template_count)):
        file_index = template_count + i
        data = np.load(open(dir_path + '/' + filenames[file_index], 'rb'))
        start_time=time.time()
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result = np.zeros((data.shape[0], data.shape[1] // 2, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1] // 2):
                result[i][j][0] = result_min_data[i, j * 2]
                result[i][j][1] = result_max_data[i, j * 2]
                result[i][j][2] = result_mean_data[i, j * 2]
                result[i][j][3] = result_min_data[i, j * 2 + 1]
                result[i][j][4] = result_max_data[i, j * 2 + 1]
                result[i][j][5] = result_mean_data[i, j * 2 + 1]
        end_time=time.time()
        duration=end_time-start_time
        key=name+'-'+filenames[file_index][:-4]
        records[key]=duration
        label_set.append(1)
        deep_label_set.append(1)
        local_dataset.append(result)
    # 计算模仿数据
    forged_dir_path = '../dataset/handwriting-lab-1/mimic-feature-chord/' + name
    forged_filenames = os.listdir(forged_dir_path)
    np.random.shuffle(forged_filenames)
    forged_filenames=forged_filenames[:45]
    for i in range(len(forged_filenames)):
        data = np.load(open(forged_dir_path + '/' + forged_filenames[i], 'rb'))
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result = np.zeros((data.shape[0], data.shape[1] // 2, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1] // 2):
                result[i][j][0] = result_min_data[i, j * 2]
                result[i][j][1] = result_max_data[i, j * 2]
                result[i][j][2] = result_mean_data[i, j * 2]
                result[i][j][3] = result_min_data[i, j * 2 + 1]
                result[i][j][4] = result_max_data[i, j * 2 + 1]
                result[i][j][5] = result_mean_data[i, j * 2 + 1]
        label_set.append(0)
        deep_label_set.append(2)
        local_dataset.append(result)
    # random forger
    randomForgerFilepaths = []

    for i in range(len(names)):
        if names[i] != name:
            dir_path = '../dataset/handwriting-lab-1/feature-chord/' + names[i]
            filenames = os.listdir(dir_path)
            np.random.shuffle(filenames)
            for i in range(4):
                filepath = os.path.join(dir_path, filenames[i])
                randomForgerFilepaths.append(filepath)
    np.random.shuffle(randomForgerFilepaths)
    randomForgerFilepaths = randomForgerFilepaths[:45]
    for filepath in randomForgerFilepaths:
        data = np.load(open(filepath, 'rb'))
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result = np.zeros((data.shape[0], data.shape[1] // 2, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1] // 2):
                result[i][j][0] = result_min_data[i, j * 2]
                result[i][j][1] = result_max_data[i, j * 2]
                result[i][j][2] = result_mean_data[i, j * 2]
                result[i][j][3] = result_min_data[i, j * 2 + 1]
                result[i][j][4] = result_max_data[i, j * 2 + 1]
                result[i][j][5] = result_mean_data[i, j * 2 + 1]
        label_set.append(0)
        deep_label_set.append(3)
        local_dataset.append(result)
    data_set += local_dataset


if __name__ == '__main__':
    main()
    # generate_dataset()