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
    frequencies = [6,7,8]
    for frequency in frequencies:
        dir_path = './frequency_model/' + str(frequency)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i in range(1, 21):
            model = get_model(frequency)
            dataset_filepath = './reference-dataset/21/dataset-{}.pkl'.format(i)
            dataset = np.load(dataset_filepath)
            train_data_set = dataset['train_data_set']
            train_data_set = np.asarray(train_data_set)
            train_data_set=train_data_set[:,:,:frequency]
            # print('shape {}'.format(train_data_set.shape))
            train_label_set = dataset['train_label_set']
            test_data_set = dataset['test_data_set']
            test_data_set = np.asarray(test_data_set)
            test_data_set=test_data_set[:,:,:frequency]
            test_label_set = dataset['test_label_set']
            model_filepath = './frequency_model/{}/{}.hdf5'.format(frequency, i)
            checkpointer = ModelCheckpoint(
                filepath=model_filepath, verbose=1,
                save_best_only=True)
            history = LossHistory()
            result = model.fit(train_data_set, np.asarray(train_label_set), batch_size=10,
                               epochs=60, verbose=1,
                               validation_data=(test_data_set, np.asarray(test_label_set)),
                               callbacks=[checkpointer, history])
            KTF.clear_session()


def get_model(frequency):
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(200, frequency, 6)))
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


if __name__ == '__main__':
    main()
    # repeat_predict(4)
    # template_count_evaluation()
