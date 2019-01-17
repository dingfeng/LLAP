# -*- coding: UTF-8 -*-
# filename: IQ_encoder_nearest date: 2018/12/19 10:34  
# author: FD 
import os
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

template_count = 12


def main():
    names = ['anna', 'chenhao', 'huangsi', 'dingfeng', 'huangsi-forged-forged', 'xuhuatao', 'yingjunhao', 'zhuyan',
             'zhangqian', 'zhaorun', 'zhuyan']
    accuracy_result = np.ones((len(names), len(names)))
    for i, name0 in enumerate(names):
        print(name0)
        for j, name1 in enumerate(names):
            if name0 == name1:
                print(1)
                continue
            records = []
            for _ in range(10):
                records += [predict(name0, name1)]
            accuracy = np.mean(records)
            accuracy_result[i, j] = accuracy
            print(accuracy)
        # print('')
    # plt.figure()
    # plt.xlabel(names)
    # plt.ylabel(names)
    # plt.plot(accuracy_result)
    # plt.show()
    pass


def predict(name0, name1):
    template0, data0 = get_data('../dataset/data20-10/cutted_IQ_features/' + name0)
    template1, data1 = get_data('../dataset/data20-10/cutted_IQ_features/' + name1)
    distance_weight = np.asarray([1, 1, 1])
    count = 0
    for data in data0:
        distances0 = get_distance_to_templates(template0, data)
        distance0 = np.mean(distances0 * distance_weight)
        distances1 = get_distance_to_templates(template1, data)
        distance1 = np.mean(distances1 * distance_weight)
        if (distance0 < distance1):
            # print('distances0 {}'.format(distances0))
            # print('distances1 {}'.format(distances1))
            # print('right')
            count += 1
    #     else:
    #         print('distances0 {}'.format(distances0))
    #         print('distances1 {}'.format(distances1))
    #         print('wrong')
    # print('right {} total {} accuracy {}'.format(count, len(data0), count / len(data0)))
    return count / len(data0)


def get_distance_to_templates(templates, data):
    min_distances = [10000, 10000, 10000]
    for template in templates:
        for features in template:
            for i in range(len(features)):
                feature = features[i].flatten()
                compared_feature = data[0][i].flatten()
                min_distances[i] = min(min_distances[i], norm(feature - compared_feature, ord=2))

    return np.asarray(min_distances)


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))
        result.append(data)
    result = shuffle(result)
    return result[:template_count], result[template_count:]


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result


if __name__ == '__main__':
    main()
