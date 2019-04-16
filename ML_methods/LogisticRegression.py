# -*- coding: UTF-8 -*-
# filename: LogisticRegression date: 2019/4/10 13:55  
# author: FD 
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
def main():
    dataset = np.load('./dataset-1-tr.pkl')
    train_data_set = np.asarray(dataset['train_data_set'])
    print(train_data_set.shape)
    train_data_set=train_data_set.reshape((train_data_set.shape[0],-1))
    train_label_set = np.asarray(dataset['train_label_set'])
    test_data_set = np.asarray(dataset['test_data_set'])
    test_data_set=test_data_set.reshape((test_data_set.shape[0],-1))
    test_label_set = np.asarray(dataset['test_label_set'])
    # clf = LogisticRegression().fit(train_data_set, train_label_set)
    # score=clf.score(test_data_set,test_label_set)
    # print('score {}'.format(score))
    # clf = SVC(gamma='auto')
    # clf.fit(train_data_set, train_label_set)
    # score=clf.score(test_data_set,test_label_set)
    # print('score {}'.format(score))
    clf = RandomForestClassifier(n_estimators=300, max_depth=50,random_state=0)
    clf.fit(train_data_set, train_label_set)
    result = clf.predict(test_data_set).ravel()
    result = np.vstack((result, np.asarray(test_label_set))).T
    fpr_total, tpr_total, thresholds_total = roc_curve(result[:, 1].astype(np.int), result[:, 0])
    AUC = auc(fpr_total, tpr_total)
    print('auc {}'.format(AUC))
    eer = brentq(lambda x: 1. - x - interp1d(fpr_total, tpr_total)(x), 0., 1.)
    print('eer {}'.format(eer))


if __name__ == '__main__':
    main()