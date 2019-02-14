# -*- coding: UTF-8 -*-
# filename: show_tcount_result date: 2019/2/2 9:41  
# author: FD 
import numpy as np
import matplotlib.pyplot as plt

def main():
    evaresult=np.load(open('./tcount_evaresult.pkl','rb'))
    # reference_amount=5
    # print('mean_auc_random={} mean_auc_mimic={} mean_auc_all={} mean_eer_random={} mean_eer_mimic={} mean_eer_all={}'.format(
    #     evaresult[reference_amount-1,0],evaresult[reference_amount-1,1],evaresult[reference_amount-1,2],evaresult[reference_amount-1,3],evaresult[reference_amount-1,4],evaresult[reference_amount-1,5]
    # ))
    # return
    # mean_auc_random, mean_auc_mimic, mean_auc_all, mean_eer_random, mean_eer_mimic, mean_eer_all
    plt.figure(figsize=(10, 6))
    # plt.subplot(211)
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,0],label='random forgers',marker='o')
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,1],label='skilled forgers',marker='*')
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,2],label='all forgers',marker='x')
    plt.xlabel('Reference Signature Amount', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.ylim(0.91, 1.0)
    plt.legend(prop = {'size':22})
    plt.tight_layout()
    plt.savefig('rcount-auc-lines.pdf', dpi=100)
    # plt.subplot(212)
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,3],label='random forgers',marker='o')
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,4],label='skilled forgers',marker='*')
    plt.plot(np.arange(1,evaresult.shape[0]+1),evaresult[:,5],label='all forgers',marker='x')
    plt.xlabel('Reference Signature Amount', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20, fontname='normal')
    plt.yticks(fontsize=20, fontname='normal')
    plt.legend(prop = {'size':22})
    # plt.ylim(0.05, 0.08)
    plt.tight_layout()
    plt.savefig('rcount-eer-lines.pdf', dpi=100)
    plt.show()
    pass


if __name__ == '__main__':
    main()