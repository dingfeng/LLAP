# -*- coding: UTF-8 -*-
# filename: svm date: 2018/12/18 23:00  
# author: FD 
import os
from scipy.linalg import norm
import numpy as np
from analysis.features import get_feature
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    data0 = get_data('../dataset/data20-10/cutted_IQ/dingfeng/')
    data1 = get_data('../dataset/data20-10/cutted_IQ/huangsi/')
    data2 = []#get_data('../dataset/data20-10/cutted_IQ/dingfeng/')+get_data('../dataset/data20-10/cutted_IQ/chenhao/')+get_data('../dataset/data20-10/cutted_IQ/yingjunhao/')+get_data('../dataset/data20-10/cutted_IQ/dengyufeng/')
    dataset = data0 + data1 + data2
    labels = [0 for i in range(len(data0))] + [1 for i in range(len(data1))] + [1 for i in range(len(data2))]
    dataset, indexes = shuffle(dataset)
    dataset = np.asarray(dataset)
    ss=MinMaxScaler()
    ss.fit(dataset)
    dataset=ss.transform(dataset)
    labels = np.asarray(labels)[indexes]
    kf = KFold(n_splits=3)
    kf.get_n_splits(dataset)  # 给出K折的折数，输出为2
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = SVC(gamma='auto',class_weight='balanced',kernel='linear')
        clf.fit(X_train, y_train)
        print('score : {}'.format(clf.score(X_test,y_test)))
        # print('scora : {}'.format(y_test))

    pass


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))[0]
        data = data - np.roll(data, 1)
        data = data[1:]
        data = norm(data, ord=2, axis=1)
        data = data - np.roll(data, 1)
        data = data[1:]
        data = np.abs(data)
        feature = get_feature(data)
        result.append(feature)
    result, _ = shuffle(result)
    return result


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result, indexes


if __name__ == '__main__':
    main()
