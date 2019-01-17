# -*- coding: UTF-8 -*-
# filename: autoencoder date: 2019/1/15 8:51  
# author: FD 
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential, Model, Input
from keras import regularizers
import keras_metrics
import numpy as np
import pickle
import os
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from analysis2.LossHistory import LossHistory
from sklearn import svm
from scipy.linalg import norm
import matplotlib.pyplot as plt
max_sequence_len = 2500

model_names = [ 'anna', 'dengyufeng','dingfeng', 'yuyinggang', 'zhangqian']

model=None

def main():
    global labels
    global model
    model=get_model()
    # labels=['dingfeng']
    # train_data,test_data= get_all_data()
    # train_data = sequence.pad_sequences(train_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    # train_data = train_data.reshape((-1, max_sequence_len // 10, 10))
    # train_result=models[0].predict(train_data)
    # for i in range(1,len(models)):
    #     train_result=np.hstack((train_result,models[i].predict(train_data)))
    # checkpointer = ModelCheckpoint(filepath="keras_autoencoder.hdf5", verbose=1, save_best_only=True)
    # history = LossHistory()
    # autoencoder_model=get_autoencoder_model()
    # autoencoder_model.fit(train_result, train_result, batch_size=50,
    #                    epochs=500, verbose=1,callbacks=[checkpointer, history])
    # autoencoder_model.save_weights("keras_autoencoder-final.hdf5")
    labels = ['jianghao']
    train_data, test_data = get_all_data()
    train_result=get_features(train_data)
    test_result=get_features(test_data)
    distances=[]
    for i in range(test_result.shape[0]):
        distances.append(get_distance(train_result,test_result[i,:]))

    labels=['jianghao-forged']
    train_data_0, test_data = get_all_data()
    train_result_1 = get_features(train_data_0)
    test_distances=[]
    for i in range(train_result_1.shape[0]):
        test_distances.append(get_distance(train_result,train_result_1[i,:]))
    plt.figure()
    # plt.scatter(distances,[1 for i in range(len(distances))],c='r')
    plt.scatter(test_distances,[1 for i in range(len(test_distances))],c='g')


def get_distance_to_template(templates,test):
    distance=1000000
    for i in range(templates.shape[0]):
        distance=min(get_distance(templates[i,:],test),distance)
    return distance

def get_distance(item0,item1):
    return norm(item0-item1,ord=2)

def get_features(data):
    global model
    train_data = sequence.pad_sequences(data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    train_data = train_data.reshape((-1, max_sequence_len // 10, 10))
    feautures=model.predict(train_data)
    return feautures

def get_model():
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_sequence_len // 10, 10)))
    model.add(Bidirectional(LSTM(128)))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l1(0.001)))
    model.load_weights('keras_total.hdf5',by_name=True)
    return model


labels = ['anna', 'anna-forged', 'dengyufeng', 'dengyufeng-forged', 'dingfeng', 'dingfeng-forged', 'yuyinggang',
          'yuyinggang-forged', 'zhangqian', 'zhangqian-forged']
test_rate = 0.5


def get_all_data():
    global labels
    dir_path = '../dataset/handwriting-lab-1/cutted'
    train_data = []
    test_data = []
    for label in os.listdir(dir_path):
        if not labels.__contains__(label):
            continue
        label_path = os.path.join(dir_path, label)
        filenames = os.listdir(label_path)
        if 'index.pkl' in filenames:
            filenames.remove('index.pkl')
        indexes = np.arange(len(filenames))
        np.random.shuffle(indexes)
        index_path = os.path.join(label_path, 'index.pkl')
        pickle.dump(indexes, open(index_path, 'wb'))
        train_top = 30  # int(len(filenames) * (1 - test_rate))
        for i in range(train_top):
            filepath = os.path.join(label_path, filenames[indexes[i]])
            data = get_data(filepath)
            train_data += data
        for i in range(train_top, 36):
            filepath = os.path.join(label_path, filenames[indexes[i]])
            data = get_data(filepath)
            test_data += data

    train_data, indexes = shuffle(train_data)
    for i in range(len(train_data)):
        train_data[i] = train_data[i] * 1e5
    for i in range(len(test_data)):
        test_data[i] = test_data[i] * 1e5
    return train_data, test_data


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
