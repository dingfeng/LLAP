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


def repeat_predict():
    dct_coefficients = [8,10,12,15,20,40,60,80,100, 120, 140, 160, 180, 200]
    results = []
    for dct_coefficient in dct_coefficients:
        total_auc_all = 0
        total_eer_all = 0
        for i in range(20):
            sess = tf.Session(config=config)
            KTF.set_session(sess)
            model_path = './only_acce/{}/{}.hdf5'.format(dct_coefficient, i + 1)
            model = get_model(dct_coefficient,model_path)
            dataset = np.load('./reference-dataset/21/dataset-{}.pkl'.format(i + 1))
            test_data_set = dataset['test_data_set']
            test_data_set = np.asarray(test_data_set)
            test_data_set=test_data_set[:,:dct_coefficient,:,[1,3,5]]
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
    return np.asarray(results)


def get_model(dct_coefficient,model_path):
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


def main():
    results = repeat_predict()
    pickle.dump(results, open('dct_acce_result.pkl', 'wb'))
    # pass


def show_evaluation_plot():
    results = np.load('dct_acce_result.pkl')
    results = np.asarray(results)
    plt.figure(figsize=(10, 6))
    plt.plot([8,10,12,15,20,40,60,80,100, 120, 140, 160, 180, 200], results[:, 0], lw=2, marker='o', c='r', markersize=12)
    plt.xlabel('Dct Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.tight_layout()
    plt.savefig('./dct_acce_auc.pdf')
    plt.figure(figsize=(10, 6))
    plt.plot([8,10,12,15,20,40,60,80,100, 120, 140, 160, 180, 200], results[:, 1], lw=2, marker='o', c='r', markersize=12)
    plt.xlabel('Dct Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.tight_layout()
    plt.savefig('./dct_acce_eer.pdf')
    plt.show()


if __name__ == '__main__':
    # main()
    # repeat_predict(4)
    # template_count_evaluation()
    show_evaluation_plot()
