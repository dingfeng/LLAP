# -*- coding: UTF-8 -*-
# filename: test date: 2019/4/8 20:41  
# author: FD 
import numpy as np


def main():
    filepath='./testdata/results/result.npz'
    result=np.load(open(filepath,'rb'))
    print(result)



if __name__ == '__main__':
    main()
