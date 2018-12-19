# -*- coding: UTF-8 -*-
# filename: generate_feature date: 2018/12/16 14:14  
# author: FD 
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import numpy as np
import os
import pickle
from scipy.linalg import norm


def main():
    convert()
    pass


def get_model():
    input_img = Input(shape=(1800, 1))
    x = Conv1D(32, 5, activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling1D(3, padding='same', name='maxpool1')(x)
    x = Conv1D(16, 5, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling1D(3, padding='same', name='maxpool2')(x)
    x = Conv1D(8, 5, activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling1D(3, padding='same', name='maxpool3')(x)
    x = Conv1D(1, 5, activation='relu', padding='same', name='conv4')(x)
    encoder = MaxPooling1D(3, padding='same', name='maxpool4')(x)
    autoencoder = Model(input_img, encoder)
    autoencoder.load_weights('auto_encoder_IQ.h5', by_name=True)
    autoencoder.summary()
    return autoencoder


def convert():
    dir_path = '../dataset/data20-10/cutted_IQ/'
    label_names = []
    dirs = os.listdir(dir_path)
    model = get_model()
    for label_name in dirs:
        label_names.append(label_name)
        label_path = os.path.join(dir_path, label_name)
        filenames = os.listdir(label_path)
        for filename in filenames:
            source = os.path.join(dir_path, label_name, filename)
            dest_dir = os.path.join('../dataset/data20-10/cutted_IQ_features', label_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, filename)
            print(source)
            print(dest)
            data_to_feature(source, dest, model)


def data_to_feature(source, dest, model):
    onedata = np.load(open(source, 'rb'))
    feature_list = []
    for data in onedata:
        next_data0, next_data1, next_data2 = get_feature_datas(data)
        feature0 = model.predict(next_data0.reshape(1, 1800, 1))
        feature1 = model.predict(next_data1.reshape(1, 1800, 1))
        feature2 = model.predict(next_data2.reshape(1, 1800, 1))
        feature_list.append([feature0, feature1, feature2])
    # print(feature_list)
    pickle.dump(feature_list, open(dest, 'wb'))


def get_feature_datas(data):
    data = data - np.roll(data, 1)
    data = data[1:]
    data0 = norm(data, ord=2, axis=1)
    data1 = data0 - np.roll(data0, 1)
    data1 = data1[1:]
    data2 = data1 - np.roll(data1, 1)
    data2 = data2[1:]
    return reshape_data(data0), reshape_data(data1), reshape_data(data2)


def reshape_data(data):
    data = normalize(data)
    # if data.shape[0] > max_len:
    #     max_len = data.shape[0]
    next_data = np.zeros(1800)
    next_data[50:len(data) + 50] = data[:]
    next_data = next_data.reshape(1800, 1)
    return next_data


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
