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
    dataset = np.load('dataset5.pkl')
    train_data_set = dataset['train_data_set']
    train_label_set = dataset['train_label_set']
    train_label_one_hot = to_categorical(train_label_set)
    test_data_set = dataset['test_data_set']
    test_label_set = dataset['test_label_set']
    test_label_one_hot = to_categorical(test_label_set)
    model = get_model()
    checkpointer = ModelCheckpoint(filepath="keras_binary_cnn.hdf5", verbose=1, save_best_only=True)
    history = LossHistory()
    result = model.fit(np.asarray(train_data_set), train_label_one_hot, batch_size=10,
                       epochs=60, verbose=1, validation_data=(np.asarray(test_data_set), test_label_one_hot),
                       callbacks=[checkpointer, history])
    print(model.metrics_names)
    print(model.evaluate(np.asarray(test_data_set),test_label_one_hot,batch_size=100))

    pass


def get_model():
    model = Sequential()

    model.add(Conv2D(64, 3, padding='same',activation='relu', input_shape=(200, 8,6)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.002),
                    activity_regularizer=regularizers.l1(0.002)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',keras_metrics.precision(label=1),keras_metrics.recall(label=1),keras_metrics.f1_score()])
    return model


if __name__ == '__main__':
    main()
