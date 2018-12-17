# -*- coding: UTF-8 -*-
# filename: encode_nearest date: 2018/12/16 15:24  
# author: FD 
import os
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def main():
    data0 = get_data('../dataset/data20-10/max_variance_cutted_features/huangsi/')
    data1 = get_data('../dataset/data20-10/max_variance_cutted_features/zhuyan/')
    labels = []
    dataset = []
    for data in data0:
        dataset.append(data)
        labels.append(0)
    for data in data1:
        dataset.append(data)
        labels.append(1)
    # dataset = np.asarray(dataset)
    # dataset = dataset.reshape(-1, 54)
    template_count = 20
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    newdataset = []
    newlabels = []
    for index in indexes:
        newdataset.append(dataset[index])
        newlabels.append(labels[index])
    train_data = newdataset[:template_count]
    train_label = newlabels[:template_count]
    test_data = newdataset[template_count:]
    test_label = newlabels[template_count:]
    other_train_data=get_other_train_data()
    train_data+=other_train_data
    train_data = np.asarray(train_data).reshape(-1, 23)
    test_data = np.asarray(test_data).reshape(-1, 23)
    clf = SVC(gamma='auto',kernel='linear', class_weight='balanced', max_iter=50)
    new_train_labels = []
    for label in train_label:
        new_train_labels.append(label)
        new_train_labels.append(label)
    for i in range(len(other_train_data)):
        new_train_labels.append(1)
        new_train_labels.append(1)
    new_test_labels = []
    for label in test_label:
        new_test_labels.append(label)
        new_test_labels.append(label)
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    clf.fit(train_data, new_train_labels)
    score = clf.score(test_data, new_test_labels)
    print(new_test_labels)
    print(score)
    pass


def get_other_train_data():
    result = []
    labels = ['anna',  'xuhuatao', 'dingfeng','zhangqian','chenhao']
    for label in labels:
        data = get_data('../dataset/data20-10/max_variance_cutted_features/' + label)
        for item in data:
            result.append(item)
    return []

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
