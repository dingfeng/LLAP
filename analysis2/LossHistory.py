# -*- coding: UTF-8 -*-
# filename: LossHistory date: 2018/12/22 22:37  
# author: FD
import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))