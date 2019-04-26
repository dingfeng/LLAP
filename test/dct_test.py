# -*- coding: UTF-8 -*-
# filename: dct_test date: 2019/4/18 15:28  
# author: FD 
import numpy as np
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler


def main():
    # coefficients=dct([)
    ss = StandardScaler()
    coefficients = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((-1, 1))
    print('np var {}'.format(np.var(coefficients, axis=0)))
    print(' var self {}'.format(np.mean(np.abs(coefficients - np.mean(coefficients)) ** 2)))
    ss.fit(coefficients)
    print('fit std {}'.format(ss.var_))
    result = ss.transform(coefficients)
    print('result {}'.format(result))
    print('my result {}'.format((coefficients - np.mean(coefficients)) / np.std(coefficients)))
    return


if __name__ == '__main__':
    main()
