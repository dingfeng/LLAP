# -*- coding: UTF-8 -*-
# filename: svm date: 2018/12/13 11:26  
# author: FD 
import os
import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from dtw import dtw

def main():
    data = readdata()
    chenhao_data = data['zhaorun']
    dengyufeng_data = data['dingfeng']
    dataset = []
    labels = []
    for i in range(len(chenhao_data)):
        I = chenhao_data[i]['I']
        Q = chenhao_data[i]['Q']
        feature = get_feature(I, Q)
        dataset.append(feature)
        labels.append(1)
    for i in range(len(dengyufeng_data)):
        I = dengyufeng_data[i]['I']
        Q = dengyufeng_data[i]['Q']
        feature = get_feature(I, Q)
        dataset.append(feature)
        labels.append(0)

    # indexes = np.arange(len(labels))
    # np.random.shuffle(indexes)
    # dataset = np.asarray(dataset)
    # labels = np.asarray(labels)
    # dataset = dataset[indexes]
    # labels = labels[indexes]
    # print('labels len {}'.format(len(labels)))
    # svc = SVC(gamma='auto')
    # svc.fit(dataset[:20], labels[:20])
    # score=svc.score(dataset[20:],labels[20:])
    # print(labels[22:])
    # print(score)
    pass


def get_feature(I, Q):
    # data =
    I = normalize(I)
    Q = normalize(Q)
    # pca = PCA(n_components=1)
    # data=np.asarray([I, Q]).T
    # pca.fit(data)
    # compressed_data = pca.transform(data).flatten()
    return np.concatenate((dct(I)[:40], dct(Q)[:40]))


def readdata():
    dir_path = '../dataset/data20-10/cutted'
    filenames = os.listdir(dir_path)
    data = {}
    for dirname in filenames:
        data_dir = os.path.join(dir_path, dirname)
        data_dir_data = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            onedata = np.load(open(filepath, 'rb'))
            # I = onedata['I']
            # Q = onedata['Q']
            # finalData = np.vstack((I, Q))
            data_dir_data.append(onedata)
        data[dirname] = data_dir_data
    return data


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    # main() dct 每个维取最小值
    main()
