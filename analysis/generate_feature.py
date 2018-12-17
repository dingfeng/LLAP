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
    autoencoder.load_weights('auto_encoder_variance.h5', by_name=True)
    autoencoder.summary()
    return autoencoder


def convert():
    dir_path = '../dataset/data20-10/max_variance_cutted/'
    label_names = []
    dirs = os.listdir(dir_path)
    model = get_model()
    for label_name in dirs:
        label_names.append(label_name)
        label_path = os.path.join(dir_path, label_name)
        filenames = os.listdir(label_path)
        for filename in filenames:
            source = os.path.join(dir_path, label_name, filename)
            dest_dir = os.path.join('../dataset/data20-10/max_variance_cutted_features', label_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, filename)
            print(source)
            print(dest)
            data_to_feature(source, dest,model)


def data_to_feature(source, dest, model):
    onedata = np.load(open(source, 'rb'))
    feature_list = []
    for data in onedata:
        data = data - np.roll(data, 1)
        data = data[1:]
        # data = data - np.roll(data, 1)
        # data = data[1:]
        data = normalize(data)
        next_data = np.zeros(1800)
        next_data[50:len(data) + 50] = data[:]
        next_data = next_data.reshape(1,1800, 1)
        feature = model.predict(next_data)
        feature_list.append(feature)
    print(feature_list)
    pickle.dump(feature_list, open(dest, 'wb'))


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    main()
