# -*- coding: UTF-8 -*-
# filename: cnn_predict date: 2019/1/28 21:29
# author: FD
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from analysis2.LossHistory import LossHistory
from keras import regularizers
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import pickle
import keras
import keras_metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配


def main():
    dct_coefficients = [8,10,12,15,20]
    for dct_coefficient in [20]:
        dir_path = './only_velocity/' + str(dct_coefficient)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i in range(1, 21):
            model = get_model(dct_coefficient)
            dataset_filepath = './dataset/dataset-{}.pkl'.format(i)
            dataset = np.load(dataset_filepath,allow_pickle=True)
            train_data_set = dataset['train_data_set']
            train_data_set = np.asarray(train_data_set)
            train_data_set=train_data_set[:,:dct_coefficient,:,[1,3,5]]
            print('shape {}'.format(train_data_set.shape))
            train_label_set = dataset['train_label_set']
            test_data_set = dataset['test_data_set']
            test_data_set = np.asarray(test_data_set)
            test_data_set=test_data_set[:,:dct_coefficient,:,[1,3,5]]
            test_label_set = dataset['test_label_set']
            model_filepath = './only_velocity/{}/{}.hdf5'.format(dct_coefficient, i)
            checkpointer = ModelCheckpoint(
                filepath=model_filepath, verbose=1,
                save_best_only=True)
            history = LossHistory()
            result = model.fit(train_data_set, np.asarray(train_label_set), batch_size=10,
                               epochs=60, verbose=1,
                               validation_data=(test_data_set, np.asarray(test_label_set)),
                               callbacks=[checkpointer, history])
            KTF.clear_session()


def get_model(dct_coefficient):
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(dct_coefficient, 8, 3)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', keras_metrics.precision(label=1), keras_metrics.recall(label=1),
                           keras_metrics.f1_score()])
    return model


def repeat_predict():
    dct_coefficients = [40]
    results = []
    for dct_coefficient in dct_coefficients:
        total_auc_all = 0
        total_eer_all = 0
        for i in range(20):
            sess = tf.Session(config=config)
            KTF.set_session(sess)
            model_path = './only_velocity/{}/{}.hdf5'.format(dct_coefficient, i + 1)
            model = get_predict_model(dct_coefficient,model_path)
            dataset = np.load('./reference-dataset/21/dataset-{}.pkl'.format(i + 1))
            test_data_set = dataset['test_data_set']
            test_data_set = np.asarray(test_data_set)
            test_data_set=test_data_set[:,:dct_coefficient,:,[0,2,4]]
            test_label_set = dataset['test_label_set']
            result = model.predict(np.asarray(test_data_set)).ravel()
            result = np.vstack((result, np.asarray(test_label_set))).T
            fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
            AUC = auc(fpr_total, tpr_total)
            total_auc_all += AUC
            print('all forger AUC {}'.format(AUC))
            eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
            total_eer_all += eer
            KTF.clear_session()
        mean_auc_all = total_auc_all / 20
        mean_eer_all = total_eer_all / 20
        print('mean_auc_all {} mean_eer_all {}'.format(mean_auc_all, mean_eer_all))
        results.append([mean_auc_all, mean_eer_all])
    print(np.asarray(results))

def get_predict_model(dct_coefficient,model_path):
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(dct_coefficient, 8, 3)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.load_weights(model_path, by_name=True)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy', keras_metrics.precision(label=1), keras_metrics.recall(label=1),
    #                        keras_metrics.f1_score()])
    return model

if __name__ == '__main__':
    main()
    # repeat_predict(4)
    # template_count_evaluation()
    # repeat_predict()
