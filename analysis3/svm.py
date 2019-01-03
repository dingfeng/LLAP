# -*- coding: UTF-8 -*-
# filename: svm date: 2019/1/3 9:52  
# author: FD 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
max_sequence_len = 2500
label2num = {'huangsi': 3, 'zhuyan': 10, }
num2label = {3: 'huangsi', 10: 'zhuyan'}
test_rate = 0.5


def main():
    model = get_model()
    train_data, train_label, test_data, test_label = get_all_data()
    train_data = sequence.pad_sequences(train_data, maxlen=max_sequence_len, padding='post', value=-2, dtype=np.float64)
    train_data = train_data.reshape((-1, max_sequence_len // 10, 10))
    test_data = sequence.pad_sequences(test_data, maxlen=max_sequence_len, padding='post', value=-2, dtype=np.float64)
    test_data = test_data.reshape((-1, max_sequence_len // 10, 10))
    train_features = model.predict(train_data)
    test_features = model.predict(test_data)
    clf = SVC(gamma='auto', class_weight='balanced', kernel='linear')
    clf.fit(train_features, train_label)
    score=clf.score(test_features,test_label)
    print('score {}'.format(score))
def get_model():
    model = Sequential()
    model.add(Masking(mask_value=-2, input_shape=(max_sequence_len // 10, 10)))
    model.add(LSTM(128, dropout=0, recurrent_dropout=0, return_sequences=True))
    model.add(LSTM(128, dropout=0, recurrent_dropout=0))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.load_weights('keras_rnn12.hdf5', by_name=True)
    print(model.summary())
    return model


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
        indexes = np.arange(len(filenames))
        np.random.shuffle(indexes)
        train_top = int(len(filenames) * (1 - test_rate))
        for i in range(train_top):
            filepath = os.path.join(label_path, filenames[i])
            data = get_data(filepath)
            train_data += data
            train_label += [label2num[label] for i in range(len(data))]
        for i in range(train_top, len(filenames)):
            filepath = os.path.join(label_path, filenames[i])
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
