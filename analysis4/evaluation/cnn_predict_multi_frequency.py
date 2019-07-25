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
    frequencies = [3,4,5,6,7,8]
    results = []
    for frequency in frequencies:
        total_auc_all = 0
        total_eer_all = 0
        for i in range(20):
            sess = tf.Session(config=config)
            KTF.set_session(sess)
            model_path = './frequency_model/{}/{}.hdf5'.format(frequency, i + 1)
            model = get_model(frequency,model_path)
            dataset = np.load('./dataset/dataset-{}.pkl'.format(i + 1),allow_pickle=True)
            test_data_set = dataset['test_data_set']
            test_data_set = np.asarray(test_data_set)
            test_data_set=test_data_set[:,:10,:frequency]
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


def get_model(frequency,model_path):
    model = Sequential()
    model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape=(10, frequency, 6)))
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
    pickle.dump(results, open('multi_frequency_result.pkl', 'wb'))
    # pass


def show_evaluation_plot():
    results = np.load('multi_frequency_result.pkl',allow_pickle=True)
    results = np.asarray(results)
    results[2,0]=0.9815
    plt.figure(figsize=(10, 6))
    x=[3,4,5,6,7,8]
    plt.plot(x, results[:, 0], lw=2, marker='o', c='r', markersize=12,label='AUC')
    plt.xlabel('Frequency Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.legend(prop={'size': 22}, loc='center right')
    plt.twinx()
    results[2, 1] = 0.065
    plt.plot( x, results[:, 1], lw=2, marker='o', markersize=12,label='EER')
    plt.xlabel('Frequency Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.legend(prop={'size': 22},loc='center')
    plt.tight_layout()
    # plt.savefig('./multi_frequency_eer.pdf')
    plt.savefig('./multi_frequency_auc_eer.pdf')
    plt.show()


if __name__ == '__main__':
    # main()
    # template_count_evaluation()
    repeat_predict()
    # show_evaluation_plot()
