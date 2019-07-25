# -*- coding: UTF-8 -*-
# filename: cnn date: 2019/1/16 14:22
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
import keras
import keras_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

def main():
    dataset = np.load('O:/evaluation2/pretrain-dataset/{}/dataset-{}.pkl'.format(10, 1), allow_pickle=True)
    train_data_set = dataset['train_data_set']
    train_data_set = np.asarray(train_data_set)
    train_data_set = train_data_set[:, :10, :]
    train_label_set = dataset['train_label_set']
    test_data_set = dataset['test_data_set']
    test_data_set = np.asarray(test_data_set)
    test_data_set = test_data_set[:, :10, :]
    test_label_set = dataset['test_label_set']
    model = get_model()
    checkpointer = ModelCheckpoint(filepath="trained-model-{}.hdf5".format(1),
                                   verbose=1, save_best_only=True)
    history = LossHistory()
    result = model.fit(np.asarray(train_data_set), np.asarray(train_label_set), batch_size=32,
                       epochs=40, verbose=2,
                       validation_data=(np.asarray(test_data_set), np.asarray(test_label_set)),
                       callbacks=[checkpointer, history])
    print(model.metrics_names)
    print(model.evaluate(np.asarray(test_data_set), np.asarray(test_label_set), batch_size=100))




def get_model():
    model = Sequential()
    model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape=(10, 8, 6)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.load_weights('model-1.hdf5', by_name=True)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', keras_metrics.precision(label=1), keras_metrics.recall(label=1),
                           keras_metrics.f1_score()])
    return model

if __name__ == '__main__':
    main()