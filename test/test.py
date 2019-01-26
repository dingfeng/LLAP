# -*- coding: UTF-8 -*-
# filename: test date: 2019/1/20 15:45  
# author: FD 
from pylab import *
styles = ['normal', 'italic', 'oblique']
weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
axis('off')
for i in range(len(styles)):
    for j in range(len(weights)):
        font={'style':styles[i],'weight':weights[j]}
        text(i*0.3,j*0.15,'Hello World',fontdict=font)
show()