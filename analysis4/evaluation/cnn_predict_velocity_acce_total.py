# -*- coding: UTF-8 -*-
# filename: cnn_predict_velocity_acce_total date: 2019/2/14 9:48  
# author: FD
import numpy as np
import matplotlib.pyplot as plt

def show_evaluation_plot():
    velocity_results = np.load('dct_velocity_result.pkl')
    velocity_results = np.asarray(velocity_results)
    acce_results=np.load('dct_acce_result.pkl')
    acce_results=np.asarray(acce_results)
    total_results=np.load('dct_result.pkl')
    total_results=np.asarray(total_results)
    ranges=[8, 10, 12, 15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    max_auc_index=np.argmax(total_results[:,0])
    min_eer_index=np.argmin(total_results[:,1])

    print('max auc {} eer {} number {} \nmin auc {} eer {} number {}'.format(total_results[max_auc_index,0],total_results[max_auc_index,1],ranges[max_auc_index],total_results[min_eer_index,0],total_results[min_eer_index,1],ranges[min_eer_index]))
    # plt.figure(figsize=(10, 6))
    # plt.plot([8,10,12,15,20,40], velocity_results[:, 0], lw=2, marker='o', markersize=12,label='velocity',color='C1')
    # plt.plot([8,10,12,15,20,40,60,80,100, 120, 140, 160, 180, 200],acce_results[:,0],lw=2, marker='o', markersize=12,label='acceleration',color='C4')
    # plt.plot([8,10,12,15,20,40,60,80,100, 120, 140, 160, 180, 200],total_results[:,0],lw=2, marker='o', markersize=12,label='both',color='C2')
    # plt.xlabel('DCT Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.xticks(fontsize=20, fontname='normal')
    # plt.yticks(fontsize=20, fontname='normal')
    # plt.legend(prop={'size': 20})
    # plt.tight_layout()
    # plt.savefig('./dct_total_auc.pdf',dpi=100)
    # plt.figure(figsize=(10, 6))
    # plt.plot([8,10,12,15,20,40], velocity_results[:, 1], lw=2, marker='o', c='r', markersize=12, label='velocity',color='C1')
    # plt.plot([8, 10, 12, 15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], acce_results[:, 1], lw=2, marker='o',
    #         markersize=12,label='acceleration',color='C4')
    # plt.plot([8, 10, 12, 15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], total_results[:, 1], lw=2, marker='o',
    #          markersize=12,label='both',color='C2')
    # plt.xlabel('DCT Coefficient Number', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    # plt.xticks(fontsize=20, fontname='normal')
    # plt.yticks(fontsize=20, fontname='normal')
    # plt.legend(prop={'size': 20})
    # plt.tight_layout()
    # plt.savefig('./dct_total_eer.pdf',dpi=100)
    # plt.show()


if __name__ == '__main__':
    show_evaluation_plot()