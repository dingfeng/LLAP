# -*- coding: UTF-8 -*-
# filename: encode_nearest date: 2018/12/16 15:24  
# author: FD 
import os
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    data0 = get_data('../dataset/data20-10/max_variance_cutted_features/dengyufeng/')
    data1 = get_data('../dataset/data20-10/max_variance_cutted_features/yuyinggang/')
    template_count = 5
    template0 = data0[:template_count]
    template1 = data1[:template_count]
    data0 = data0[template_count:]
    data1 = data1[template_count]
    count = 0
    # print('shape of template 0 {}'.format(template0[0].shape))
    # template0 = get_template_by_cluster(template0, 4)
    # template1 = get_template_by_cluster(template1, 4)
    for data in data0:
        distance0 = get_distance(template0, data)
        distance1 = get_distance(template1, data)
        print('distance0 {} distance1 {}'.format(distance0, distance1))
        if (distance0 < distance1):
            print('right')
            count += 1
        else:
            print('wrong')
    print('count {} / total {} accuracy   {}'.format(count, len(data0), count / len(data0)))
    pass


def test_forge():
    data0 = get_data('../dataset/data20-10/features/dingfeng/')
    template_count = 15
    template = data0[:template_count]
    mimic_dir_path = '../dataset/data20-10-mimic/features'
    # filepath_list = []
    plt.figure()
    plt.title('euclidean distance')
    for filename in os.listdir(mimic_dir_path):
        filepath = os.path.join(mimic_dir_path, filename)
        distance_list = get_distance_list(template, filepath)
        plt.scatter(distance_list, [0 for i in range(len(distance_list))], label=filename)
    real_distance_list = get_distance_list_data(template, data0[template_count:])
    plt.scatter(real_distance_list, [0 for i in range(len(data0[template_count:]))], label='real distance')
    plt.legend()
    plt.show()


def get_template_by_cluster(data, template_count):
    data = np.asarray(data).reshape(-1, 107)
    cls = KMeans(n_clusters=template_count, init='k-means++')
    cls.fit(data)
    centers = cls.cluster_centers_
    result = []
    for i in range(centers.shape[0]):
        center = centers[i, :].reshape(107, -1)
        result.append(center)
    return result


def get_distance_list(template, filepath):
    data = get_data(filepath)
    return get_distance_list_data(template, data)


def get_distance_list_data(template, data):
    result = []
    for data0 in data:
        distance0 = get_distance(template, data0)
        result.append(distance0)
    return result


def get_distance(data0, data1):
    min_distance = 10000
    # for data in data0:
    for templates in data0:
        for template in templates:
            for compared in data1:
                distance = norm(template.flatten() - compared.flatten(), ord=2)
                if distance < min_distance:
                    min_distance = distance

    return min_distance


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))
        result.append(data)
    result = shuffle(result)
    return result


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result


if __name__ == '__main__':
    main()
    # test_forge()
