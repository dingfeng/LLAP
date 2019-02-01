# -*- coding: UTF-8 -*-
# filename: test date: 2019/1/29 11:37  
# author: FD 
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.bar('skilled forgers', 2,ec='r', ls='--', lw=2,color='C0')
plt.bar('random forgers', 3,ec='r', ls='--', lw=2,color='C1')
plt.bar('all forgers', 4,ec='r', ls='--', lw=2,color='C2')
plt.xlabel('Forger Types', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
plt.ylabel('AUC', fontdict={'style': 'normal', 'weight': 'bold', 'size': 22})
plt.xticks(fontsize=17, fontname='normal')
plt.yticks(fontsize=17, fontname='normal')
plt.tight_layout()
plt.show()
