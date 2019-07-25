# -*- coding: UTF-8 -*-
# filename: generate_dataset date: 2019/1/16 13:30
# author: FD
import numpy as np
import os
import pickle

data_set = []
label_set = []
deep_label_set=[]
names = None


def generate_dataset():
    global names
    global data_set
    global label_set
    global deep_label_set
    names=os.listdir('../../dataset/handwriting-lab-3/forged-feature-chord-40')
    for reference_amount in [10]:
        for k in range(1):
            data_set = []
            label_set = []
            deep_label_set = []
            for name in names:
                generate_by_name(name,reference_amount)
            indexes = np.arange(len(data_set))
            np.random.shuffle(indexes)
            test_rate = 0.2
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
            dir_path='O:/evaluation2/pretrain-dataset/{}'.format(reference_amount)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            pickle.dump(
                {'train_data_set': train_data_set, 'train_label_set': train_label_set, 'test_data_set': test_data_set,
                 'test_label_set': test_label_set,'deep_test_label_set':deep_test_label_set}, open('O:/evaluation2/pretrain-dataset/{}/dataset-{}.pkl'.format(reference_amount,k + 1), 'wb'))

def generate_by_name(name,template_count):
    global names
    global data_set
    global deep_label_set
    dir_path = '../../dataset/handwriting-lab-3/feature-chord-40/' + name
    filenames = os.listdir(dir_path)
    # print('name {}'.format())
    # print('len of filenames {}'.format(len(filenames)))
    # filenames=filenames[:50]
    # indexes = np.arange(len(filenames))
    np.random.shuffle(filenames)
    # filenames = filenames[:110]
    # template_count = 20
    templates = []
    for i in range(template_count):
        data = np.load(open(dir_path + '/' + filenames[i], 'rb'),allow_pickle=True)
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
    for i in range(min(20,len(filenames)-template_count)):
        file_index = template_count + i
        data = np.load(open(dir_path + '/' + filenames[file_index], 'rb'),allow_pickle=True)
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
                # for k in range(len(max_record)):
                #     result[i][j][k]=np.abs(result[i][j][k])
                #     max_record[k] = max(result[i][j][k], max_record[k])

                # result[i][j][6] = result_sum_data[i, j * 2]
                # result[i][j][7] = result_sum_data[i, j * 2 + 1]
                # result[i][j][3]=result_sum_data[i,j]
        label_set.append(1)
        deep_label_set.append(1)
        local_dataset.append(result)
    # 计算模仿数据
    forged_dir_path = '../../dataset/handwriting-lab-3/forged-feature-chord-40/'+name
    forged_filenames = os.listdir(forged_dir_path)
    np.random.shuffle(forged_filenames)
    forged_filenames=forged_filenames[:10]
    for i in range(len(forged_filenames)):
        data = np.load(open(forged_dir_path + '/' + forged_filenames[i], 'rb'),allow_pickle=True)
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
                # for k in range(len(max_record)):
                #     result[i][j][k] = np.abs(result[i][j][k])
                #     max_record[k] = max(result[i][j][k], max_record[k])
        label_set.append(0)
        deep_label_set.append(2)
        local_dataset.append(result)
    # random forger
    randomForgerFilepaths = []
    indexes=list(range(len(names)))
    np.random.shuffle(indexes)
    random_names=[]
    for i in indexes:
        random_names.append(names[i])
    random_names.remove(name)
    for i in range(5):
        if random_names[i] != name:
            dir_path = '../../dataset/handwriting-lab-3/feature-chord-40/' + random_names[i]
            filenames = os.listdir(dir_path)
            np.random.shuffle(filenames)
            for i in range(2):
                filepath = os.path.join(dir_path, filenames[i])
                randomForgerFilepaths.append(filepath)
    for filepath in randomForgerFilepaths:
        data = np.load(open(filepath, 'rb'),allow_pickle=True)
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
        label_set.append(0)
        deep_label_set.append(3)
        local_dataset.append(result)
    data_set += local_dataset


if __name__ == '__main__':
    # main()
    generate_dataset()