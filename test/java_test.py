# -*- coding: UTF-8 -*-
# filename: java_test date: 2019/4/20 19:30  
# author: FD 
import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Flatten, Conv2D, MaxPooling2D, BatchNormalization
def main():
    file_content=open('features.json','r').readline()
    features=np.asarray(json.loads(file_content))
    # for feature in features:
    #     feature=np.asarray(feature)
    #     print(feature.shape)
    model = get_model('lab2_model.hdf5')
    music_result = model.predict(features)
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
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.load_weights(model_path, by_name=True)
    return model

if __name__ == '__main__':
    main()