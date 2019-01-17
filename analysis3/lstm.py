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
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import regularizers
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

max_sequence_len = 2500


def main():
    # width=5
    train_data, train_label, test_data, test_label = get_all_data()
    train_data = sequence.pad_sequences(train_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    train_data = train_data.reshape((-1, max_sequence_len, 1))
    train_label_one_hot = to_categorical(train_label)
    test_data = sequence.pad_sequences(test_data, maxlen=max_sequence_len, padding='post', value=0, dtype=np.float64)
    test_data = test_data.reshape((-1, max_sequence_len, 1))
    test_label_one_hot = to_categorical(test_label)
    checkpointer = ModelCheckpoint(filepath="keras_one_person_cnn.hdf5", verbose=1, save_best_only=True)
    history = LossHistory()
    model = get_cnn_model()
    # result = model.fit(train_data, train_label_one_hot, batch_size=50,
    #                    epochs=100, verbose=1, validation_data=(test_data, test_label_one_hot),
    #                    callbacks=[checkpointer, history])
    # print(model.evaluate(train_data,train_label_one_hot,batch_size=100))
    predicted_result=model.predict(train_data)
    print('result')
    # model.save('keras_rnn_epochend5.hdf5')
    return


import keras_metrics
def get_model():
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_sequence_len // 10, 10)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                 activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dropout(0.3))
    model.add(Dense(14, activation='softmax'))
    # model.summary()
    model.load_weights("keras_one_person_cnn1.hdf5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def get_cnn_model():
    model = Sequential()
    model.add(Conv1D(128, 8, activation='relu', input_shape=(2500, 1)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(32, activation='relu', ))
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001),
                     activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dropout(0.2))
    model.add(Dense(12, activation='softmax'))
    print(model.summary())
    model.load_weights("keras_one_person_cnn.hdf5",by_name=True)
    model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
# label2num = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2, 'huangsi': 3, 'qingpeijie': 4, 'xuhuatao': 5, 'yinjunhao': 6,
#              'yuyinggang': 7, 'zhangqian': 8, 'zhaorun': 9, 'zhuyan': 10, 'jianghao': 11, 'chenhao': 12, 'chenbo': 13,
#              'dingfeng-forged':14,'dengyufeng-forged':15,'anna-forged':16,'huangsi-forged':17,'qingpeijie-forged':18,
#              'xuhuatao-forged':19,'yinjunhao-forged':20,'yuyinggang-forged':21,'zhangqian-forged':22,'zhaorun-forged':23,
#              'zhuyan-forged':24,'jianghao-forged':25,'chenhao-forged':26,'chenbo-forged':27}
# label2num = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2, 'huangsi': 3, 'qingpeijie': 4, 'xuhuatao': 5, 'yinjunhao': 6,
#              'yuyinggang': 7, 'zhangqian': 8, 'zhaorun': 9, 'zhuyan': 10, 'jianghao': 11,
#              'dingfeng-forged':0,'dingyufeng-forged':1,'anna-forged':2,'huangsi-forged':3,'qingpeijie-forged':4,'xuhuatao-forged':5,'yinjunhao':6,
#              'yuyinggang-forged':7,'zhangqian-forged':8,'zhaorun-forged':9,'zhuyan-forged':10,'jianghao-forged':11}
label2num = {'dingfeng': 0}
# label2num = {'anna': 0, 'anna-forged':1,'dingfeng':2,'dingfeng-forged':3,'zhangqian':4,'zhangqian-forged':5,
#              'yuyinggang':6,'yuyinggang-forged':7,'dengyufeng':8,'dengyufeng-forged':9}
# num2label = {0: 'dingfeng', 1: 'dengyufeng', 2: 'anna', 3: 'huangsi', 4: 'qingpeijie', 5: 'xuhuatao', 6: 'yinjunhao',
#              7: 'yuyinggang', 8: 'zhangqian', 9: 'zhaorun', 10: 'zhuyan', 11: 'jianghao', 12: 'chenhao', 13: 'chenbo'
#              ,14:'dingfeng-forged',15:'dengyufeng-forged',16:'anna-forged',17:'huangsi-forged',18:'qingpeijie-forged',
#              19:'xuhuatao-forged',20:'yinjunhao-forged',21:'yuyinggang-forged',22:'zhangqian-forged',23:'zhaorun-forged',
#              24:'zhuyan-forged',25:'jianghao-forged',26:'chenhao-forged',27:'chenbo-forged'}
# label2num = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2,'huangsi':3,'qingpeijie':4,'xuhuatao':5}
# num2label = {0: 'dingfeng', 1: 'dengyufeng', 2: 'anna',3:'huangsi',4:'qingpeijie',5:'xuhuatao'}
test_rate = 0.01


def get_all_data():
    dir_path = '../dataset/handwriting-lab-1/mimic_cutted_arranged2/'
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    seq_max_len = -1
    for label in os.listdir(dir_path):
        # if  not label.endswith('forged'):
        #     continue
        if not label2num.keys().__contains__(label):
            continue
        label_path = os.path.join(dir_path, label)
        filenames = os.listdir(label_path)
        if 'index.pkl' in filenames:
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
        for i in range(train_top, (len(filenames))):
            filepath = os.path.join(label_path, filenames[indexes[i]])
            print(filepath)
            data = get_data(filepath)
            test_data += data
            test_label += [label2num[label] for i in range(len(data))]

    # train_data, indexes = shuffle(train_data)
    # train_label = np.asarray(train_label)[indexes].tolist()
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
