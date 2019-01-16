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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

def main():
    dataset = np.load('dataset.pkl')
    train_data_set = dataset['train_data_set']
    train_label_set = dataset['train_label_set']
    train_label_one_hot = to_categorical(train_label_set)
    test_data_set = dataset['test_data_set']
    test_label_set = dataset['test_label_set']
    test_label_one_hot = to_categorical(test_label_set)
    model = get_model()
    checkpointer = ModelCheckpoint(filepath="keras_one_person_cnn.hdf5", verbose=1, save_best_only=True)
    history = LossHistory()
    result = model.fit(np.asarray(train_data_set), train_label_one_hot, batch_size=50,
                       epochs=100, verbose=1, validation_data=(np.asarray(test_data_set), test_label_one_hot),
                       callbacks=[checkpointer, history])
    pass


def get_model():
    model = Sequential()

    model.add(Conv2D(128, 3, padding='same',activation='relu', input_shape=(200, 8,3)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=50, dim=(200,8), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = ID#np.load('data/' + ID + '.npy')
            # Store class
            y[i] = self.labels[i]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    main()
