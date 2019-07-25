# -*- coding: UTF-8 -*-
# filename: generate_dataset date: 2019/4/16 15:07  
# author: FD

# name: anna
#  day1 0.6708551645278931
#  day7 0.9343920648097992
# name: dingfeng
#  day1 0.9398826956748962
#  day7 0.9539699256420135
# name: jianghao
#  day1 0.9442923963069916
#  day7 0.966691255569458
# name: zhuyan
#  day1 0.8458971083164215
#  day7 0.9005178809165955

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

# keys = ['music', 'walk', '0day','1day','7day'
#         ]
# keys = ['distance0', 'distance1', 'distance2']
# keys = ['replayattack', 'replayattack-1']
keys = ['0day','1day','7day']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)


# keys=['reference']

def main():
    # show_replay_result()
    show_robustness_result()

def show_replay_result():
    dataset_dir = '../replay-attack-dataset'
    names = os.listdir(dataset_dir)
    model = get_model('1.hdf5')
    final_results = []
    for name in names:
        if name == 'dengyufeng':
            continue
        filepath = '../replay-attack-dataset/{}/feature.pkl'.format(name)
        results = get_dataset(filepath)
        # music_result = model.predict(results['music'])
        # walk_result = model.predict(results['walk'])
        # day_0_result = model.predict(results['0day'])
        # day_1_result = model.predict(results['1day'])
        # day_2_result = model.predict(results['7day'])
        print('name: {}'.format(name))
        # print_result(model, results)
        final_results += model.predict(results['replayattack-1']).ravel().tolist()
    print('median max {} {}'.format(np.median(final_results), np.max(final_results)))
    return

def show_robustness_result():
    dataset_dir = '../dataset2'
    names = os.listdir(dataset_dir)
    names.remove('dengyufeng')
    model = get_model('1.hdf5')
    for name in names:
        filepath = '../dataset2/{}/feature.pkl'.format(name)
        results = get_dataset(filepath)
        print('name: {} '.format(name))
        day0_results = model.predict(results['1day']).ravel().tolist()
        print(' day1 {} '.format(np.median(day0_results)))
        day7_results = model.predict(results['7day']).ravel().tolist()
        print(' day7 {} '.format(np.median(day7_results)))
    return


def print_result(model, results):
    for key in keys:
        result = model.predict(results[key])
        print('median max {} : {} {}'.format(key, np.median(result),np.max(result)))


def get_dataset(filepath):
    feature_file = np.load(open(filepath, 'rb'), allow_pickle=True)
    # references = feature_file['reference']
    # references = feature_file['distance2']
    references = feature_file['0day']
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
        dataset = np.asarray(dataset)
        dataset = dataset[:, :10, :]
        results[key] = dataset
    return results


def get_model(model_path):
    model = Sequential()
    model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape=(10, 8, 6)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 3, activation='relu'))
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
