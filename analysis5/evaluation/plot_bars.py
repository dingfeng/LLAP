# -*- coding: UTF-8 -*-
# filename: plot_bars date: 2019/2/3 15:00  
# author: FD 
import matplotlib.pyplot as plt
import numpy as np


def main():
    template_list = [6, 8, 10, 12]
    data_list = []
    for template in template_list:
        filename = 'template-{}-result.pkl'.format(template)
        data = np.load('./predict-result/' + filename)
        data_list.append(data)
    data_list=np.asarray(data_list)
    total_width, n = 0.9, 3
    width = total_width / n
    x = np.arange(4)
    plt.figure(figsize=(10, 10))
    plt.bar(x, data_list[:,0],width=width,ec='r', ls='--', lw=2, color='C0',label='random forgers')
    plt.bar(x+width, data_list[:, 1],width=width,tick_label = template_list, ec='r', ls='--', lw=2, color='C1',label='skilled forgers')
    plt.bar(x + width*2, data_list[:, 2],width=width, ec='r', ls='--', lw=2, color='C2',label='all forgers')
    print('auc {}'.format(data_list[:,2]))
    plt.xlabel('Reference Signature Amount', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.85, 1.0)
    plt.legend(prop={'size': 22})
    plt.tight_layout()
    plt.savefig('cross-user-auc-bars.pdf')

    plt.figure(figsize=(10, 10))
    plt.bar(x, data_list[:, 3], width=width, ec='r', ls='--', lw=2, color='C0', label='random forgers')
    plt.bar(x + width, data_list[:, 4], width=width, tick_label=template_list, ec='r', ls='--', lw=2, color='C1',
            label='skilled forgers')
    plt.bar(x + width * 2, data_list[:, 5], width=width, ec='r', ls='--', lw=2, color='C2', label='all forgers')
    print('eer {}'.format(data_list[:,5]))
    plt.xlabel('Reference Signature Amount', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.ylabel('EER', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size': 22})
    plt.tight_layout()
    plt.savefig('cross-user-eer-bars.pdf')
    plt.show()


if __name__ == '__main__':
    main()
