# -*- coding: UTF-8 -*-
# filename: generate_dataset date: 2019/4/16 15:07  
# author: FD 
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from analysis2.LossHistory import LossHistory
from keras import regularizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
keys = ['music', 'walk', '0day','1day','7day']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
# keys=['reference']

def main():
    filepath = '../dataset2/zhuyan/feature.pkl'
    results = get_dataset(filepath)
    model = get_model('lab2_model.hdf5')
    music_result = model.predict(results['music'])
    walk_result = model.predict(results['walk'])
    day_0_result = model.predict(results['0day'])
    day_1_result = model.predict(results['1day'])
    day_2_result = model.predict(results['7day'])
    print_result(model, results)
    return


def print_result(model, results):
    for key in keys:
        result = model.predict(results[key])
        print('mean {} : {}'.format(key, np.median(result)))


def get_dataset(filepath):
    feature_file = np.load(filepath, 'rb')
    # references = feature_file['reference']
    references=feature_file['walk']
    shape = (200, 16)
    min_data = np.zeros(shape)
    max_data = np.zeros(shape)
    mean_data = np.zeros(shape)
    sum_data = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            total = 0
            min_value = 100000
            max_value = -100000
            for template in references:
                total += template[i][j]
                min_value = min(min_value, template[i][j])
                max_value = max(max_value, template[i][j])
            min_data[i][j] = min_value
            max_data[i][j] = max_value
            sum_data[i][j] = total
            mean_data[i][j] = total / len(references)
    results = {}
    for key in keys:
        dataset = []
        datas = feature_file[key]
        for i in range(len(datas)):
            data = datas[i]
            result_min_data = data - min_data
            result_max_data = data - max_data
            result_mean_data = data - mean_data
            result = np.zeros((data.shape[0], data.shape[1] // 2, 6))
            for i in range(data.shape[0]):
                for j in range(data.shape[1] // 2):
                    result[i][j][0] = result_min_data[i, j * 2]
                    result[i][j][1] = result_max_data[i, j * 2]
                    result[i][j][2] = result_mean_data[i, j * 2]
                    result[i][j][3] = result_min_data[i, j * 2 + 1]
                    result[i][j][4] = result_max_data[i, j * 2 + 1]
                    result[i][j][5] = result_mean_data[i, j * 2 + 1]
            dataset.append(result)
        results[key] = np.asarray(dataset)
    return results


def get_model(model_path):
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(200, 8, 6)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.load_weights(model_path, by_name=True)
    return model


if __name__ == '__main__':
    main()
