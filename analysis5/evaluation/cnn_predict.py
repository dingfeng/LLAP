# -*- coding: UTF-8 -*-
# filename: cnn_predict date: 2019/1/28 21:29  
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
import pickle
import keras
import keras_metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配


def main():
    model_path = 'O:/evaluation/reference-model/21/model-19.hdf5'
    # model_path = './model/model-1.hdf5'
    model = get_model(model_path)
    dataset = np.load('O:/evaluation/reference-dataset/21/dataset-19.pkl')
    # dataset=np.load('./dataset/dataset-1.pkl')
    test_data_set = dataset['test_data_set']
    test_label_set = dataset['test_label_set']
    # 'deep_test_label_set': deep_test_label_set}
    deep_test_label_set = np.asarray(dataset['deep_test_label_set'])

    mimic_indices = np.where(deep_test_label_set != 2)[0]
    result = model.predict(np.asarray(test_data_set)[mimic_indices]).ravel()
    result = np.vstack((result, np.asarray(test_label_set)[mimic_indices])).T
    fpr_random, tpr_random, thresholds_random = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_random, tpr_random)
    print('random forger AUC {}'.format(AUC))
    eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
    # thresh = interp1d(fpr_random, thresholds_random)(eer)
    print('random forger err {}'.format(eer))

    random_indices = np.where(deep_test_label_set != 3)[0]
    result = model.predict(np.asarray(test_data_set)[random_indices]).ravel()
    result = np.vstack((result, np.asarray(test_label_set)[random_indices])).T
    fpr_mimic, tpr_mimic, thresholds_mimic = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_mimic, tpr_mimic)
    print('mimic forger AUC {}'.format(AUC))
    eer = brentq(lambda x: 1. - x - interp1d(fpr_mimic, tpr_mimic)(x), 0., 1.)
    # thresh = interp1d(fpr_mimic, thresholds_mimic)(eer)
    print('mimic forger err {}'.format(eer))

    result = model.predict(np.asarray(test_data_set)).ravel()
    result = np.vstack((result, np.asarray(test_label_set))).T
    fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_total, tpr_total)
    print('total forger AUC {}'.format(AUC))
    eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
    # thresh = interp1d(fpr_total, thresholds_total)(eer)
    print('total forger err {}'.format(eer))
    # plt.plot(fpr, tpr, lw=2)
    plt.figure(figsize=(5, 5))
    plt.xlabel('False Positive Rate', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('True Positive Rate', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')

    plt.tight_layout()
    plt.plot(fpr_random, tpr_random, label='random', lw=2)
    plt.plot(fpr_mimic, tpr_mimic, label='skilled', lw=2)
    plt.plot(fpr_total, tpr_total, label='all', lw=2)
    plt.plot([0,1],[1,0],linestyle="--")
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.legend(prop={'size': 20})
    plt.savefig('mimic-random-total-ROC-curves.pdf', dpi=100)
    plt.show()
    pass


def repeat_predict(template_count):
    total_auc_random = 0
    total_auc_mimic = 0
    total_auc_all = 0
    total_eer_random=0
    total_eer_mimic=0
    total_eer_all=0
    count=1
    for j in range(20):
        for i in range(10):
            print('count {}'.format(count))
            count+=1
            sess = tf.Session(config=config)
            KTF.set_session(sess)
            model_path = 'O:/evaluation/reference-model/21/model-{}.hdf5'.format(j+1)
            model = get_model(model_path)
            dataset = np.load('O:/evaluation/reference-dataset/{}/dataset-{}.pkl'.format(template_count,i+1))
            test_data_set = dataset['test_data_set']
            test_label_set = dataset['test_label_set']
            # 'deep_test_label_set': deep_test_label_set}
            deep_test_label_set = np.asarray(dataset['deep_test_label_set'])

            mimic_indices = np.where(deep_test_label_set != 2)[0]
            result = model.predict(np.asarray(test_data_set)[mimic_indices]).ravel()
            result = np.vstack((result, np.asarray(test_label_set)[mimic_indices])).T
            fpr_random, tpr_random, thresholds_random = roc_curve(result[:, 1].astype(np.int), result[:, 0])
            AUC = auc(fpr_random, tpr_random)
            total_auc_random += AUC
            eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
            total_eer_random+=eer
            # print('random forger AUC {} eer {}'.format(AUC,eer))
            random_indices = np.where(deep_test_label_set != 3)[0]
            result = model.predict(np.asarray(test_data_set)[random_indices]).ravel()
            result = np.vstack((result, np.asarray(test_label_set)[random_indices])).T
            fpr_mimic, tpr_mimic, thresholds_mimic = roc_curve(result[:, 1].astype(np.int), result[:, 0])
            AUC = auc(fpr_mimic, tpr_mimic)
            total_auc_mimic += AUC
            eer = brentq(lambda x: 1. - x - interp1d(fpr_mimic, tpr_mimic)(x), 0., 1.)
            total_eer_mimic+=eer
            # print('mimic forger AUC {} eer {}'.format(AUC,eer))

            result = model.predict(np.asarray(test_data_set)).ravel()
            result = np.vstack((result, np.asarray(test_label_set))).T
            fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
            AUC = auc(fpr_total, tpr_total)
            total_auc_all += AUC
            eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
            total_eer_all+=eer
            # print('all forger AUC {} eer {}'.format(AUC,eer))
            # plt.plot(fpr, tpr, lw=1)
            # plt.show()
            KTF.clear_session()


    mean_auc_random = total_auc_random / 200
    mean_auc_mimic = total_auc_mimic / 200
    mean_auc_all = total_auc_all / 200
    print('mean_auc_random={} mean_auc_mimic={} mean_auc_all={}'.format(mean_auc_random,mean_auc_mimic,mean_auc_all))
    # plt.figure(figsize=(5, 5))
    # plt.bar('skilled', mean_auc_mimic,ec='r', ls='--', lw=2,color='C0')
    # plt.bar('random', mean_auc_random,ec='r', ls='--', lw=2,color='C1')
    # plt.bar('all', mean_auc_all,ec='r', ls='--', lw=2,color='C2')
    # plt.xlabel('Forger Types', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.xticks(fontsize=20, fontname='normal')
    # plt.yticks(fontsize=20, fontname='normal')
    # plt.ylim(0.91,1.0)
    # plt.tight_layout()
    # plt.savefig('auc-bars.pdf', dpi=100)
    # plt.show()
    mean_eer_random=total_eer_random/200
    mean_eer_mimic=total_eer_mimic/200
    mean_eer_all=total_eer_all/200
    # print('mean_eer_random={} mean_eer_mimic={} mean_eer_all={}'.format(mean_eer_random, mean_eer_mimic,
    #                                                                     mean_eer_all))
    # plt.figure(figsize=(5, 5))
    # plt.bar('skilled', mean_eer_mimic, ec='r', ls='--', lw=2, color='C0')
    # plt.bar('random', mean_eer_random, ec='r', ls='--', lw=2, color='C1')
    # plt.bar('all', mean_eer_all, ec='r', ls='--', lw=2, color='C2')
    # plt.xlabel('Forger Types', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.xticks(fontsize=20, fontname='normal')
    # plt.yticks(fontsize=20, fontname='normal')
    # plt.ylim(0.05, 0.08)
    # plt.tight_layout()
    # plt.savefig('eer-bars.pdf', dpi=100)
    # plt.show()
    return np.asarray([mean_auc_random,mean_auc_mimic,mean_auc_all,mean_eer_random,mean_eer_mimic,mean_eer_all])

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


def template_count_evaluation():
    template_evaluation_result=[]
    for i in range(22):
        repeat_predict_result=repeat_predict(i+1)
        template_evaluation_result.append(repeat_predict_result)
    template_evaluation_result=np.asarray(template_evaluation_result)
    pickle.dump(template_evaluation_result,open('tcount_evaresult.pkl','wb'))

if __name__ == '__main__':
    # main()
    for i in [6,8,10,12]:
        result=repeat_predict(i)
        pickle.dump(result,open('./predict-result/template-{}-result.pkl'.format(i),'wb'))
        print('mean_auc_random={} mean_auc_mimic={} mean_auc_all={} mean_eer_random={} mean_eer_mimic={} mean_eer_all={}'.format(result[0],result[1],result[2],result[3],result[4],result[5]))
    # template_count_evaluation()
