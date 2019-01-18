# -*- coding: UTF-8 -*-
# filename: generate_dataset date: 2019/1/16 13:30  
# author: FD 
import numpy as np
import os
import pickle
data_set = []
label_set = []
names = None

def main():
    global names
    names=['anna','chenbo',
           'chenhao',
           'dengyufeng','dingfeng','huangsi','jianghao','qingpeijie','xuhuatao','yinjunhao','yuyinggang','zhangqian','zhaorun','zhuyan']
    for name in names:
        generate_by_name(name)
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
                 'test_label_set': test_label_set}, open('dataset5.pkl', 'wb'))

def generate_by_name(name):
    global names
    dir_path = '../dataset/handwriting-lab-1/feature3/' + name
    filenames = os.listdir(dir_path)
    # filenames=filenames[:50]
    indexes = np.arange(len(filenames))
    np.random.shuffle(indexes)
    template_count = 20
    templates = []
    for i in range(template_count):
        data = np.load(open(dir_path + '/' + filenames[indexes[i]], 'rb'))
        templates.append(data)
    min_data = np.zeros(data.shape)
    max_data = np.zeros(data.shape)
    mean_data = np.zeros(data.shape)
    sum_data=np.zeros(data.shape)
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
            sum_data[i][j]=total
            mean_data[i][j] = total / len(templates)
    for i in range(len(filenames) - template_count):
        file_index = template_count + i
        data = np.load(open(dir_path + '/' + filenames[indexes[file_index]], 'rb'))
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result_sum_data=data-sum_data
        result = np.zeros((data.shape[0], data.shape[1]//2,6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]//2):
                result[i][j][0] = result_min_data[i, j*2]
                result[i][j][1] = result_max_data[i, j*2]
                result[i][j][2]= result_mean_data[i, j*2]
                result[i][j][3] = result_min_data[i, j*2+1]
                result[i][j][4] = result_max_data[i, j*2+1]
                result[i][j][5] = result_mean_data[i, j*2+1]
                # result[i][j][6] = result_sum_data[i, j * 2]
                # result[i][j][7] = result_sum_data[i, j * 2 + 1]
                # result[i][j][3]=result_sum_data[i,j]
        label_set.append(1)
        data_set.append(result)
    # 计算模仿数据
    forged_dir_path = '../dataset/handwriting-lab-1/forged-feature3/'+name
    forged_filenames = os.listdir(forged_dir_path)
    # forged_filenames=forged_filenames[:40]
    for i in range(len(forged_filenames)):
        data = np.load(open(forged_dir_path + '/' + forged_filenames[i], 'rb'))
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result_sum_data=data-sum_data
        result = np.zeros((data.shape[0], data.shape[1]//2,6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]//2):
                result[i][j][0] = result_min_data[i, j * 2]
                result[i][j][1] = result_max_data[i, j * 2]
                result[i][j][2] = result_mean_data[i, j * 2]
                result[i][j][3] = result_min_data[i, j * 2 + 1]
                result[i][j][4] = result_max_data[i, j * 2 + 1]
                result[i][j][5] = result_mean_data[i, j * 2 + 1]
                # result[i][j][6] = result_sum_data[i, j * 2]
                # result[i][j][7] = result_sum_data[i, j * 2+1]
                # result[i][j][3]=result_sum_data[i,j]
        label_set.append(0)
        data_set.append(result)
    #random forger
    randomForgerFilepaths=[]
    for i in range(len(names)):
        if names[i] != name:
            dir_path='../dataset/handwriting-lab-1/feature3/' + names[i]
            filenames=os.listdir(dir_path)
            np.random.shuffle(filenames)
            for i in range(3):
                filepath=os.path.join(dir_path,filenames[i])
                randomForgerFilepaths.append(filepath)
    for filepath in randomForgerFilepaths:
        data = np.load(open(filepath, 'rb'))
        result_min_data = data - min_data
        result_max_data = data - max_data
        result_mean_data = data - mean_data
        result_sum_data = data - sum_data
        result = np.zeros((data.shape[0], data.shape[1] // 2, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1] // 2):
                result[i][j][0] = result_min_data[i, j * 2]
                result[i][j][1] = result_max_data[i, j * 2]
                result[i][j][2] = result_mean_data[i, j * 2]
                result[i][j][3] = result_min_data[i, j * 2 + 1]
                result[i][j][4] = result_max_data[i, j * 2 + 1]
                result[i][j][5] = result_mean_data[i, j * 2 + 1]
                # result[i][j][6] = result_sum_data[i, j * 2]
                # result[i][j][7] = result_sum_data[i, j * 2+1]
                # result[i][j][3]=result_sum_data[i,j]
        label_set.append(0)
        data_set.append(result)


if __name__ == '__main__':
    main()
