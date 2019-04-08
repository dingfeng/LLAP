# -*- coding: UTF-8 -*-
# filename: ShowForgedMapping date: 2019/4/8 19:13  
# author: FD 
import numpy as np
import os
def main():
    dir_path='./testdata/forged'
    names=os.listdir(dir_path)
    dict1={}
    dict2={}
    for i in range(len(names)):
        dict1[i]=names[i]
        dict2[names[i]]=i
    print('dict 1 {}'.format(str(dict1)))
    print('dict 2 {}'.format(str(dict2)))



if __name__ == '__main__':
    main()