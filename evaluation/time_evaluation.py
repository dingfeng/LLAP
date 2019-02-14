# -*- coding: UTF-8 -*-
# filename: time_evaluation date: 2019/2/6 21:15  
# author: FD 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pylab import *
import matplotlib


def main():
    data = np.load('time_evaluation.pkl')
    raw_times = data['raw_times']
    down_sampling_times = data['down_sampling_times']
    feature_collection_times = data['feature_collection_times']
    plt.figure(figsize=(10, 5))
    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.title('down sampling times')
    plt.xlabel(r'\textbf{Ton (Seconds)}', fontsize=22)
    plt.ylabel(r'\textbf{T1 (Seconds)}', fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.scatter(raw_times, down_sampling_times)
    x = np.asarray(raw_times)
    y = np.asarray(down_sampling_times)
    m, b = polyfit(x, y, 1)
    print('m = {} b = {}'.format(m, b))
    plt.plot([0, max(x)], m * np.asarray([0, max(x)]) + b, '--k')
    plt.xlim(0)
    plt.tight_layout()
    # plt.figure()
    # plt.title('feature collection times')
    # plt.scatter(raw_times,feature_collection_times)
    plt.savefig('T1-Ton.pdf')
    plt.show()
    # feature_collection_times=np.asarray(feature_collection_times)+0.00039426622882721916
    # print('mean val {}'.format(np.mean(feature_collection_times)))
    # print('variance val {}'.format(np.std(feature_collection_times)))


def save_data_series():
    feature_time_records = np.load('../analysis4/feature_time_records')
    generate_dataset_records = np.load('../analysis4/generate_dataset_records')
    records = np.load('../analysis3/records')
    raw_times = []
    down_sampling_times = []
    feature_collection_times = []
    for key in generate_dataset_records.keys():
        raw_times.append(records[key][0])
        down_sampling_times.append(records[key][1])
        feature_collection_times.append(feature_time_records[key] + generate_dataset_records[key])
    pickle.dump({'raw_times': raw_times, 'down_sampling_times': down_sampling_times,
                 'feature_collection_times': feature_collection_times}, open('time_evaluation.pkl', 'wb'))


def off_line_time():
    data = np.load('time_evaluation.pkl')
    down_sampling_times = data['down_sampling_times']
    feature_collection_times = data['feature_collection_times']
    result=np.asarray(down_sampling_times)+np.asarray(feature_collection_times)+0.00037807593585322143
    print('mean {} std {}'.format(np.mean(result),np.std(result)))

if __name__ == '__main__':
    # main()
    # save_data_series()
    off_line_time()
