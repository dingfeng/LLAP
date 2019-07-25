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
    model_names = ['random forest','svm',' naive bayes']
    model_funs = [#get_RF_Model,
                  get_svm_model,
                  #get_naive_bayes_model
                 ]
    results = []
    for i in range(len(model_funs)):
        result = process(model_funs[i])
        print('model {} result {}'.format(model_names[i], str(result)))
        results.append(result)
    pickle.dump(results, open('result.pkl', 'wb'))


def process(model_fun):
    # model_path = './model/model-1.hdf5'
    total_auc_random = 0
    total_auc_mimic = 0
    total_auc_all = 0
    total_eer_random=0
    total_eer_mimic=0
    total_eer_all=0
    count=1
    for i in range(count):
        dataset = np.load('O:/evaluation2/reference-dataset/20/dataset-{}.pkl'.format(i+1),allow_pickle=True)
        train_data_set = np.asarray(dataset['train_data_set'])
        train_data_set = train_data_set[:, :10, :]
        train_data_set = train_data_set.reshape((train_data_set.shape[0], -1))
        train_label_set = np.asarray(dataset['train_label_set'])
        model = model_fun(train_data_set, train_label_set)
        test_data_set = np.asarray(dataset['test_data_set'])
        test_data_set = test_data_set[:, :10, :]
        test_data_set = test_data_set.reshape((test_data_set.shape[0], -1))
        test_label_set = np.asarray(dataset['test_label_set'])
        # 'deep_test_label_set': deep_test_label_set}
        deep_test_label_set = np.asarray(dataset['deep_test_label_set'])
        process_result = []
        mimic_indices = np.where(deep_test_label_set != 2)[0]
        result = model.predict_proba(test_data_set[mimic_indices]).ravel()
        result=result[np.arange(1,len(result),step=2)]
        result = np.vstack((result, test_label_set[mimic_indices])).T
        fpr_random, tpr_random, thresholds_random = roc_curve(result[:, 1].astype(np.int), result[:, 0])
        AUC = auc(fpr_random, tpr_random)
        total_auc_random+=AUC
        eer = brentq(lambda x: 1. - x - interp1d(fpr_random, tpr_random)(x), 0., 1.)
        total_eer_random+=eer
        random_indices = np.where(deep_test_label_set != 3)[0]
        result = model.predict_proba(test_data_set[random_indices]).ravel()
        result = result[np.arange(1, len(result), step=2)]
        result = np.vstack((result, test_label_set[random_indices])).T
        fpr_mimic, tpr_mimic, thresholds_mimic = roc_curve(result[:, 1].astype(np.int), result[:, 0])
        AUC = auc(fpr_mimic, tpr_mimic)
        total_auc_mimic += AUC
        eer = brentq(lambda x: 1. - x - interp1d(fpr_mimic, tpr_mimic)(x), 0., 1.)
        total_eer_mimic+=eer
        result = model.predict_proba(test_data_set).ravel()
        result = result[np.arange(1, len(result), step=2)]
        result = np.vstack((result, test_label_set)).T
        fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
        AUC = auc(fpr_total, tpr_total)
        total_auc_all+=AUC
        eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
        total_eer_all+=eer
    total_auc_random /=count
    total_auc_mimic /= count
    total_auc_all /= count
    total_eer_random /= count
    total_eer_mimic /= count
    total_eer_all /= count
    return [total_auc_random,total_auc_mimic,total_auc_all,total_eer_random,total_eer_mimic,total_eer_all]



def get_RF_Model(train_set, train_label):
    train_data_set = train_set.reshape((train_set.shape[0], -1))
    train_label_set = train_label
    clf = RandomForestClassifier(n_estimators=300, max_depth=50, random_state=0)
    clf.fit(train_data_set, train_label_set)
    return clf


def get_svm_model(train_set, train_label):
    train_data_set = train_set.reshape((train_set.shape[0], -1))
    train_label_set = train_label
    clf = SVC(gamma='scale',probability=True)
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
