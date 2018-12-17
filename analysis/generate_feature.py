# -*- coding: UTF-8 -*-
# filename: generate_feature date: 2018/12/16 14:14  
# author: FD 
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import numpy as np
import os
import pickle


def main():

    convert()
    pass


def get_model():
    input_img = Input(shape=(1700, 1))
    x = Conv1D(32, 3, activation='relu', padding='same', name='conv1')(input_img)
    x = MaxPooling1D(2, padding='same', name='maxpool1')(x)
    x = Conv1D(16, 3, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling1D(2, padding='same', name='maxpool2')(x)
    x = Conv1D(16, 3, activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling1D(2, padding='same', name='maxpool3')(x)
    x = Conv1D(1, 3, activation='relu', padding='same', name='conv5')(x)
    encoder = MaxPooling1D(2, padding='same', name='maxpool5')(x)
    autoencoder = Model(input_img, encoder)
    autoencoder.load_weights('auto_encoder.h5', by_name=True)
    autoencoder.summary()
    return autoencoder


def convert():
    dir_path = '../dataset/data20-10-mimic/cutted/'
    label_names = []
    dirs = os.listdir(dir_path)
    model = get_model()
    for label_name in dirs:
        label_names.append(label_name)
        label_path = os.path.join(dir_path, label_name)
        filenames = os.listdir(label_path)
        for filename in filenames:
            source = os.path.join(dir_path, label_name, filename)
            dest_dir = os.path.join('../dataset/data20-10/features', label_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, filename)
            print(source)
            print(dest)
            data_to_feature(source, dest,model)


def data_to_feature(source, dest, model):
    onedata = np.load(open(source, 'rb'))['I']
    feature_list = []
    for data in onedata:
        data = data - np.roll(data, 1)
        data = data[1:]
        data = data - np.roll(data, 1)
        data = data[1:]
        data = normalize(data)
        data.resize(1700, 1)
        data=data.reshape((1,1700,1))
        feature = model.predict(data)
        feature_list.append(feature)
    print(feature_list)
    pickle.dump(feature_list, open(dest, 'wb'))


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
