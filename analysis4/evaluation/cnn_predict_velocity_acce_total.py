# -*- coding: UTF-8 -*-
# filename: cnn_predict_velocity_acce_total date: 2019/2/14 9:48  
# author: FD
import numpy as np
import matplotlib.pyplot as plt

def show_evaluation_plot():
    velocity_results = np.load('dct_velocity_result.pkl',allow_pickle=True)
    velocity_results = np.asarray(velocity_results)
    acce_results=np.load('dct_acce_result.pkl',allow_pickle=True)
    acce_results=np.asarray(acce_results)
    total_results=np.load('dct_result.pkl',allow_pickle=True)
    total_results=np.asarray(total_results)
    max_auc_index=np.argmax(total_results[:,0])
    min_eer_index=np.argmin(total_results[:,1])
    plt.figure(figsize=(10, 6))
    x=[8,9,10, 15, 20, 25, 30, 35, 40]
    plt.plot(x, velocity_results[:, 0], lw=2, marker='o', markersize=12,label='velocity',color='C1')
    plt.plot(x,acce_results[:,0],lw=2, marker='o', markersize=12,label='acceleration',color='C4')
    plt.plot(x,total_results[:,0],lw=2, marker='o', markersize=12,label='both',color='C2')
    plt.xlabel('DCT Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig('./dct_total_auc.pdf',dpi=100)
    plt.figure(figsize=(10, 6))
    plt.plot(x, velocity_results[:, 1], lw=2, marker='o', c='r', markersize=12, label='velocity',color='C1')
    plt.plot(x, acce_results[:, 1], lw=2, marker='o',
            markersize=12,label='acceleration',color='C4')
    plt.plot(x, total_results[:, 1], lw=2, marker='o',
             markersize=12,label='both',color='C2')
    plt.xlabel('DCT Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig('./dct_total_eer.pdf',dpi=100)
    plt.show()


if __name__ == '__main__':
    show_evaluation_plot()
