# -*- coding: UTF-8 -*-
# filename: lstm date: 2018/12/14 19:22  
# author: FD
import os
from keras import Sequential
from keras.layers import LSTM, Masking, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
from keras.utils import to_categorical
import numpy as np
from keras import regularizers
from scipy.linalg import norm


def main():
    train()
    pass


def train():
    train_data, train_labels, test_data, test_labels = get_data()
    indexes = np.arange(train_data.shape[0])
    np.random.shuffle(indexes)
    train_data = train_data[indexes, :, :]
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_train_labels = one_hot_train_labels[indexes]
    one_hot_test_labels = to_categorical(test_labels)
    model = get_model()
    model.fit(train_data, one_hot_train_labels, batch_size=100, validation_data=(test_data, one_hot_test_labels),
              epochs=1000)

    pass


def get_model():
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=(1700, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))
    # print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def get_data():
    dir_path = '../dataset/data20-10/cutted_IQ/'
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    count = 0
    label_names = []
    max_len = 0
    dirs = os.listdir(dir_path)
    # dirs=['zhuyan','dingfeng-zhangqian']
    for label_name in dirs:
        label_names.append(label_name)
        label_path = os.path.join(dir_path, label_name)
        filenames = os.listdir(label_path)
        filenames_len = len(filenames)
        indexes = np.arange(filenames_len)
        np.random.shuffle(indexes)
        train_size = 17
        for i in range(train_size):
            index = indexes[i]
            filename = filenames[index]
            filepath = os.path.join(label_path, filename)
            onedata = np.load(open(filepath, 'rb'))
            for data in onedata:
                data = data - np.roll(data, 1)
                data = data[1:]
                data = norm(data, ord=2, axis=1)
                data = data - np.roll(data, 1)
                data[0] = 0
                data = data - np.roll(data, 1)
                data[0] = 0
                data = np.abs(data)
                data = normalize(data)
                if data.shape[0] > max_len:
                    max_len = data.shape[0]
                data = data.reshape(-1, 1)
                data = np.vstack(
                    (np.zeros((50, 1)), data, np.zeros((1700 - data.shape[0] - 50, 1))))
                train_data.append(data)
                train_labels.append(count)
        for i in range(train_size, len(indexes)):
            index = indexes[i]
            filename = filenames[index]
            filepath = os.path.join(label_path, filename)
            onedata = np.load(open(filepath, 'rb'))
            for data in onedata:
                data = data - np.roll(data, 1)
                data = data[1:]
                data = norm(data, ord=2, axis=1)
                data = data - np.roll(data, 1)
                data[0] = 0
                data = data - np.roll(data, 1)
                data[0] = 0
                data=np.abs(data)
                data = normalize(data)
                if data.shape[0] > max_len:
                    max_len = data.shape[0]
                data = data.reshape(-1, 1)
                data = np.vstack(
                    (np.zeros((50, 1)), data, np.zeros((1700 - data.shape[0] - 50, 1))))
                # data.resize(3000, 1)
                test_data.append(data)
                test_labels.append(count)
        count += 1
    print("labelnames {}".format(label_names))
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    print('max_len {}'.format(max_len))
    return train_data, train_labels, test_data, test_labels


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
