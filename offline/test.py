# -*- coding: UTF-8 -*-
# filename: test date: 2019/4/8 20:41  
# author: FD 
import numpy as np


def main():
    filepath='./testdata/results/result.npz'
    result=np.load(open(filepath,'rb'),allow_pickle=True)
    print(str(result)[:2000])



if __name__ == '__main__':
    main()
