# -*- coding: UTF-8 -*-
# filename: auto-encoder date: 2018/12/16 12:23  
# author: FD 
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
import numpy as np
import os


def main():
    train()
    pass


def train():
    train_data, train_labels, test_data, test_labels = get_data()
    autoencoder = get_model()
    autoencoder.fit(train_data, train_data, epochs=100, batch_size=128,
                    shuffle=True, validation_data=(test_data, test_data), verbose=1)
    autoencoder.save_weights('auto_encoder_variance.h5')
    pass


def get_model():
    input_img = Input(shape=(1800, 1))
    x = Conv1D(16, 5, activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling1D(3, padding='same', name='maxpool1')(x)
    x = Conv1D(8, 5, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling1D(3, padding='same', name='maxpool2')(x)
    x = Conv1D(4, 5, activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling1D(3, padding='same', name='maxpool3')(x)
    x = Conv1D(1, 5, activation='relu', padding='same', name='conv4')(x)
    encoder = MaxPooling1D(3, padding='same', name='maxpool4')(x)
    x = Conv1D(1, 5, activation='relu', padding='same')(encoder)
    x = UpSampling1D(3)(x)
    x = Conv1D(4, 5, activation='relu', padding='same')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(8, 5, activation='relu', padding='valid')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(16, 5, activation='relu', padding='valid')(x)
    x = UpSampling1D(3)(x)
    decoded = Conv1D(1, 16, activation='sigmoid', padding='valid')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.summary()
    return autoencoder


def get_data():
    dir_path = '../dataset/data20-10/max_variance_cutted/'
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    count = 0
    label_names = []
    max_len = 0
    dirs = os.listdir(dir_path)
    for label_name in dirs:
        label_names.append(label_name)
        label_path = os.path.join(dir_path, label_name)
        filenames = os.listdir(label_path)
        filenames_len = len(filenames)
        indexes = np.arange(filenames_len)
        np.random.shuffle(indexes)
        train_size = 18
        for i in range(train_size):
            index = indexes[i]
            filename = filenames[index]
            filepath = os.path.join(label_path, filename)
            onedata = np.load(open(filepath, 'rb'))
            for data in onedata:
                data = data - np.roll(data, 1)
                data = data[1:]
                data = data - np.roll(data, 1)
                data = data[1:]
                data = normalize(data)
                if data.shape[0] > max_len:
                    max_len = data.shape[0]
                next_data = np.zeros(1800)
                next_data[50:len(data) + 50] = data[:]
                # data.resize(3000, 1)
                next_data=next_data.reshape(1800,1)
                train_data.append(next_data)
                train_labels.append(count)
        for i in range(train_size, len(indexes)):
            print(i)
            index = indexes[i]
            filename = filenames[index]
            filepath = os.path.join(label_path, filename)
            onedata = np.load(open(filepath, 'rb'))
            for data in onedata:
                data = data - np.roll(data, 1)
                data = data[1:]
                data = data - np.roll(data, 1)
                data = data[1:]
                data = normalize(data)
                if data.shape[0] > max_len:
                    max_len = data.shape[0]
                next_data = np.zeros(1800)
                next_data[50:len(data)+50] = data[:]
                next_data = next_data.reshape(1800, 1)
                # data.resize(3000, 1)
                test_data.append(next_data)
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
