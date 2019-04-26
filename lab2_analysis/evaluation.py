# -*- coding: UTF-8 -*-
# filename: evaluation date: 2019/4/16 15:05  
# author: FD 
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from analysis2.LossHistory import LossHistory
from keras import regularizers


def main():
    pass

def get_model(model_path):
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(200, 8, 6)))
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