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
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

# keys = ['music', 'walk', '0day','1day','7day'
#         ]
# keys = ['distance0', 'distance1', 'distance2']
# keys = ['replayattack', 'replayattack-1']
keys = ['0day', '1day', '7day']
names = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)


# keys=['reference']

def main():
    show_robustness_result()


def show_robustness_result():
    global names
    dataset_dir = '../dataset2'
    names = os.listdir(dataset_dir)
    names.remove('dengyufeng')
    model = get_model('1.hdf5')
    labels = [1 for i in range(20)] + [0 for i in range(20)]
    for name in names:
        filepath = '../dataset2/{}/feature.pkl'.format(name)
        print(name)
        results = get_dataset(filepath, name)
        day_0_result = model.predict(results['1day']).ravel()
        # day_0_result = model.predict(results['distance0']).ravel()
        day_0_result = np.vstack((day_0_result, np.asarray(labels))).T
        fpr_random, tpr_random, thresholds_random = roc_curve(day_0_result[:, 1].astype(np.int), day_0_result[:, 0])
        AUC = auc(fpr_random, tpr_random)
        print('day 1 random forger AUC {}'.format(AUC))
        eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
        print('day 1 random forger err {}'.format(eer))

        day_7_result = model.predict(results['7day']).ravel()
        # day_7_result = model.predict(results['distance1']).ravel()
        day_7_result = np.vstack((day_7_result, np.asarray(labels))).T
        fpr_random, tpr_random, thresholds_random = roc_curve(day_7_result[:, 1].astype(np.int), day_7_result[:, 0])
        AUC = auc(fpr_random, tpr_random)
        print('day 7 random forger AUC {}'.format(AUC))
        eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
        print('day 7 random forger err {}'.format(eer))
    # print('median max {} {}'.format(np.median(final_results), np.max(final_results)))

    return


def print_result(model, results):
    for key in keys:
        result = model.predict(results[key])
        print('median max {} : {} {}'.format(key, np.median(result), np.max(result)))


def get_dataset(filepath, name):
    min_data, max_data, mean_data = get_reference_param(filepath)
    dataset = get_dataset_by_reference(filepath, min_data, max_data, mean_data)
    for key in keys:
        dataset[key] = dataset[key][:20, :, :, :]
    other_dataset = None
    for other_name in names:
        if name == other_name:
            continue
        other_filepath = '../dataset2/{}/feature.pkl'.format(other_name)
        current_other_dataset = get_dataset_by_reference(other_filepath, min_data, max_data, mean_data)
        if other_dataset is None:
            other_dataset = current_other_dataset
        else:
            other_dataset = merge2dataset(other_dataset, current_other_dataset)

    indexes = np.arange(other_dataset['1day'].shape[0])
    # indexes = np.arange(other_dataset['distance1'].shape[0])
    np.random.shuffle(indexes)
    indexes = indexes[:20]
    for key in keys:
        other_dataset[key] = other_dataset[key][indexes, :, :, :]
    dataset = merge2dataset(dataset, other_dataset)
    return dataset


def merge2dataset(dataset_one, dataset_two):
    for key in keys:
        dataset_one[key] = np.vstack((dataset_one[key], dataset_two[key]))
    return dataset_one


def get_reference_param(filepath):
    feature_file = np.load(open(filepath, 'rb'), allow_pickle=True)
    references = feature_file['0day']
    # references = feature_file['distance2']
    shape = (200, 16)
    min_data = np.zeros(shape)
    max_data = np.zeros(shape)
    mean_data = np.zeros(shape)
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
            mean_data[i][j] = total / len(references)
    return min_data, max_data, mean_data


def get_dataset_by_reference(filepath, min_data, max_data, mean_data):
    feature_file = np.load(open(filepath, 'rb'), allow_pickle=True)
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
