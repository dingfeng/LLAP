# -*- coding: UTF-8 -*-
# filename: mimic_queue date: 2019/1/6 21:15  
# author: FD 
import numpy as np

mapping = {'dingfeng': 0, 'dengyufeng': 1, 'anna': 2, 'huangsi': 3, 'qingpeijie': 4, 'xuhuatao': 5, 'yinjunhao': 6,
           'yuyinggang': 7, 'zhangqian': 8, 'zhaorun': 9, 'zhuyan': 10, 'jianghao': 11, 'chenhao': 12, 'chenbo': 13}


def main():
    names = list(mapping.keys())
    indexes = np.arange(len(names))
    np.random.shuffle(indexes)
    print(np.asarray(names)[np.asarray(indexes)])
    pass


['jianghao', 'huangsi', 'qingpeijie', 'anna', 'xuhuatao', 'yuyinggang', 'dingfeng', 'yinjunhao', 'zhangqian',
 'chenhao','zhuyan', 'chenbo', 'dengyufeng']
if __name__ == '__main__':
    main()
