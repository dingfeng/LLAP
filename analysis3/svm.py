# -*- coding: UTF-8 -*-
# filename: svm date: 2019/1/3 9:52  
# author: FD 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
import os
from sklearn.svm import SVC
from sklearn import svm

max_sequence_len = 2500
label2num = {'huangsi': 3, 'zhuyan': 10, }
num2label = {3: 'huangsi', 10: 'zhuyan'}
test_rate = 0.5


def main():
    global label2num
    global num2label
    global test_rate
    # model = get_model()
    label2num = {'yinjunhao': 3}
    num2label = {3: 'yinjunhao'}
    train_features, test_features = get_train_test_features()
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf")
    clf.fit(train_features)
    true_predict = clf.predict(test_features)
    true_predict += 1
    true_predict = true_predict / 2
    true_predict = true_predict.reshape((-1, 40))
    true_predict = np.sum(true_predict, axis=1) // 20
    nozero_count = np.count_nonzero(true_predict)
    print('true_predict shape  {} nozero {} accuracy {}'.format(true_predict.shape, nozero_count,
                                                                nozero_count / len(true_predict)))

    # for label in ['dingfeng', 'dengyufeng', 'anna','huangsi','qingpeijie','xuhuatao','yinjunhao','yuyinggang','zhangqian','zhaorun','zhuyan','jianghao']:
    for label in ['chenhao']:
        print('label {}'.format(label))
        label2num = {label: 6}
        num2label = {6: label}
        test_rate = 1
        train_features, test_features = get_train_test_features()
        true_predict = clf.predict(test_features)
        true_predict += 1
        true_predict = true_predict / 2
        true_predict = true_predict.reshape((-1, 40))
        true_predict = np.sum(true_predict, axis=1) // 20
        nozero_count = np.count_nonzero(true_predict)
        print('false_predict shape  {} nozero {} accuracy {}'.format(true_predict.shape, nozero_count,
                                                                     nozero_count / len(true_predict)))


def get_train_test_features():
    global label2num
    global num2label
    train_data, train_label, test_data, test_label = get_all_data()
    train_data = sequence.pad_sequences(train_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    train_data = train_data.reshape((-1, max_sequence_len // 10, 10))
    test_data = sequence.pad_sequences(test_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    test_data = test_data.reshape((-1, max_sequence_len // 10, 10))
    train_features = get_features(train_data)
    test_features = get_features(test_data)
    return train_features, test_features


def get_features(data):
    return


# def get_model():
#     model = Sequential()
#     model.add(Masking(mask_value=-2, input_shape=(max_sequence_len // 10, 10)))
#     model.add(LSTM(128, dropout=0, recurrent_dropout=0, return_sequences=True))
#     model.add(LSTM(128, dropout=0, recurrent_dropout=0))
#     model.add(BatchNormalization())
#     model.add(Dense(64, activation='relu'))
#     model.load_weights('keras_rnn12.hdf5', by_name=True)
#     print(model.summary())
#     return model


def get_all_data():
    global label2num
    global num2label
    global test_rate
    dir_path = '../dataset/handwriting-lab-1/cutted'
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    seq_max_len = -1
    for label in os.listdir(dir_path):
        if not label2num.keys().__contains__(label):
            continue
        label_path = os.path.join(dir_path, label)
        filenames = os.listdir(label_path)
        indexes = np.arange(len(filenames))
        np.random.shuffle(indexes)
        train_top = int(len(filenames) * (1 - test_rate))
        for i in range(train_top):
            filepath = os.path.join(label_path, filenames[i])
            data = get_data(filepath)
            train_data += data
            train_label += [label2num[label] for i in range(len(data))]
        for i in range(train_top, len(filenames)):
            filepath = os.path.join(label_path, filenames[i])
            data = get_data(filepath)
            test_data += data
            test_label += [label2num[label] for i in range(len(data))]

    train_data, indexes = shuffle(train_data)
    train_label = np.asarray(train_label)[indexes].tolist()
    max_value = 0
    for data in train_data:
        max_value = max(np.max(np.abs(data)), max_value)
        seq_max_len = max(len(data), seq_max_len)
    for data in test_data:
        max_value = max(np.max(np.abs(data)), max_value)
        seq_max_len = max(len(data), seq_max_len)
    print('seq max len {}'.format(seq_max_len))
    for i in range(len(train_data)):
        train_data[i] = train_data[i] * 1e5
    for i in range(len(test_data)):
        test_data[i] = test_data[i] * 1e5

    print('seq_max_len {}'.format(seq_max_len))
    return train_data, train_label, test_data, test_label


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result, indexes


def get_data(filepath):
    data = np.load(open(filepath, 'rb'))
    return data


if __name__ == '__main__':
    main()
