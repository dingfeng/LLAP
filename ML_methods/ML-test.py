# -*- coding: UTF-8 -*-
# filename: ML-test date: 2019/4/10 14:40  
# author: FD 
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pickle


def main():
    model_names = ['random_forest', 'svm', 'naive bayes']
    model_funs = [get_RF_Model, get_svm_model, get_naive_bayes_model]
    results = []
    for i in range(len(model_funs)):
        result = process(model_funs[i])
        print('model {} result {}'.format(model_names[i], str(result)))
        results.append(result)
    pickle.dump(results, open('result.pkl', 'wb'))


def process(model_fun):
    # model_path = './model/model-1.hdf5'
    dataset = np.load('O:/evaluation/reference-dataset/21/dataset-19.pkl')
    # dataset=np.load('./dataset/dataset-1.pkl')
    train_data_set = np.asarray(dataset['train_data_set'])
    train_data_set = train_data_set.reshape((train_data_set.shape[0], -1))
    train_label_set = np.asarray(dataset['train_label_set'])
    model = model_fun(train_data_set, train_label_set)

    test_data_set = np.asarray(dataset['test_data_set'])
    test_data_set = test_data_set.reshape((test_data_set.shape[0], -1))
    test_label_set = np.asarray(dataset['test_label_set'])
    # 'deep_test_label_set': deep_test_label_set}
    deep_test_label_set = np.asarray(dataset['deep_test_label_set'])
    process_result = []
    mimic_indices = np.where(deep_test_label_set != 2)[0]
    result = model.predict(test_data_set[mimic_indices]).ravel()
    result = np.vstack((result, test_label_set[mimic_indices])).T
    fpr_random, tpr_random, thresholds_random = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_random, tpr_random)
    # print('random forger AUC {}'.format(AUC))
    process_result.append(AUC)
    eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
    # thresh = interp1d(fpr_random, thresholds_random)(eer)
    # print('random forger err {}'.format(eer))
    process_result.append(eer)
    random_indices = np.where(deep_test_label_set != 3)[0]
    result = model.predict(test_data_set[random_indices]).ravel()
    result = np.vstack((result, test_label_set[random_indices])).T
    fpr_mimic, tpr_mimic, thresholds_mimic = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_mimic, tpr_mimic)
    # print('mimic forger AUC {}'.format(AUC))
    process_result.append(AUC)
    eer = brentq(lambda x: 1. - x - interp1d(fpr_mimic, tpr_mimic)(x), 0., 1.)
    # thresh = interp1d(fpr_mimic, thresholds_mimic)(eer)
    # print('mimic forger err {}'.format(eer))
    process_result.append(eer)
    result = model.predict(test_data_set).ravel()
    result = np.vstack((result, test_label_set)).T
    fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_total, tpr_total)
    # print('total forger AUC {}'.format(AUC))
    process_result.append(AUC)
    eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
    # thresh = interp1d(fpr_total, thresholds_total)(eer)
    # print('total forger err {}'.format(eer))
    process_result.append(eer)
    return process_result



def get_RF_Model(train_set, train_label):
    train_data_set = train_set.reshape((train_set.shape[0], -1))
    train_label_set = train_label
    clf = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=0)
    clf.fit(train_data_set, train_label_set)
    return clf


def get_svm_model(train_set, train_label):
    train_data_set = train_set.reshape((train_set.shape[0], -1))
    train_label_set = train_label
    clf = SVC(gamma='auto')
    clf.fit(train_data_set, train_label_set)
    return clf


def get_naive_bayes_model(train_set, train_label):
    train_data_set = train_set.reshape((train_set.shape[0], -1))
    train_label_set = train_label
    gnb = GaussianNB()
    gnb.fit(train_data_set, train_label_set)
    return gnb


if __name__ == '__main__':
    main()
