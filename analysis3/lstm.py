# -*- coding: UTF-8 -*-
# filename: lstm date: 2018/12/22 21:19
# author: FD
from keras.preprocessing import sequence
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from analysis2.LossHistory import LossHistory
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import pickle
from keras import regularizers
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

max_sequence_len = 2500


def main():
    train_data, train_label, test_data, test_label = get_all_data()
    train_data = sequence.pad_sequences(train_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    train_data = train_data.reshape((-1, max_sequence_len // 10, 10))
    train_label_one_hot = to_categorical(train_label)
    test_data = sequence.pad_sequences(test_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    test_data = test_data.reshape((-1, max_sequence_len // 10, 10))
    test_label_one_hot = to_categorical(test_label)

    checkpointer = ModelCheckpoint(filepath="keras_rnn14-half-test.hdf5", verbose=1, save_best_only=True, )
    history = LossHistory()
    model = get_model()
    result = model.fit(train_data, train_label_one_hot, batch_size=100,
                       epochs=100, verbose=1, validation_data=(test_data, test_label_one_hot),
                       callbacks=[checkpointer, history])
    print(model.evaluate(test_data, test_label_one_hot, batch_size=50))
    model.save('keras_rnn_epochend5.hdf5')
    return


def get_model():
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_sequence_len // 10, 10)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(BatchNormalization())
    # model.add(Dense(32, activation='relu',))
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(len(label2num), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


label2num = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2, 'huangsi': 3, 'qingpeijie': 4, 'xuhuatao': 5, 'yinjunhao': 6,
             'yuyinggang': 7, 'zhangqian': 8, 'zhaorun': 9, 'zhuyan': 10, 'jianghao': 11, 'chenhao': 12, 'chenbo': 13}
num2label = {0: 'dingfeng', 1: 'dengyufeng', 2: 'anna', 3: 'huangsi', 4: 'qingpeijie', 5: 'xuhuatao', 6: 'yinjunhao',
             7: 'yuyinggang', 8: 'zhangqian', 9: 'zhaorun', 10: 'zhuyan', 11: 'jianghao', 12: 'chenhao', 13: 'chenbo'}
# label2num = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2,'huangsi':3,'qingpeijie':4,'xuhuatao':5}
# num2label = {0: 'dingfeng', 1: 'dengyufeng', 2: 'anna',3:'huangsi',4:'qingpeijie',5:'xuhuatao'}
test_rate = 0.5


def get_all_data():
    dir_path = '../dataset/handwriting-lab-1/cutted'
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    seq_max_len = -1
    for label in os.listdir(dir_path):
        if not label2num.keys().__contains__(label):
            continue
        label_path = os.path.join(dir_path, label)
        filenames = os.listdir(label_path)
        filenames.remove('index.pkl')
        indexes = np.arange(len(filenames))
        np.random.shuffle(indexes)
        index_path=os.path.join(label_path,'index.pkl')
        pickle.dump(indexes,open(index_path,'wb'))
        train_top = int(len(filenames) * (1 - test_rate))
        for i in range(train_top):
            filepath = os.path.join(label_path, filenames[indexes[i]])
            data = get_data(filepath)
            train_data += data
            train_label += [label2num[label] for i in range(len(data))]
        for i in range(train_top, len(filenames)):
            filepath = os.path.join(label_path, filenames[indexes[i]])
            print(filepath)
            data = get_data(filepath)
            test_data += data
            test_label += [label2num[label] for i in range(len(data))]

    train_data, indexes = shuffle(train_data)
    train_label = np.asarray(train_label)[indexes].tolist()
    max_value = 0
    for data in train_data:
        max_value = max(np.max(np.abs(data)), max_value)
        seq_max_len = max(len(data), seq_max_len)
    for data in test_data:
        max_value = max(np.max(np.abs(data)), max_value)
        seq_max_len = max(len(data), seq_max_len)
    print('seq max len {}'.format(seq_max_len))
    for i in range(len(train_data)):
        train_data[i] = train_data[i] * 1e5
    for i in range(len(test_data)):
        test_data[i] = test_data[i] * 1e5

    print('seq_max_len {}'.format(seq_max_len))
    return train_data, train_label, test_data, test_label


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result, indexes


def get_data(filepath):
    data = np.load(open(filepath, 'rb'))
    return data


if __name__ == '__main__':
    main()
