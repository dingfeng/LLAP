# -*- coding: UTF-8 -*-
# filename: encode_nearest date: 2018/12/16 15:24  
# author: FD 
import os
import numpy as np
from scipy.linalg import norm


def main():
    data0 = get_data('../dataset/data20-10/features/chenhao/')
    data1 = get_data('../dataset/data20-10/features/dingfeng/')
    template_count = 10
    template0 = data0[:template_count]
    template1 = data1[:template_count]
    data0 = data0[template_count:]
    data1 = data1[template_count]
    count = 0
    for data in data0:
        distance0 = get_distance(template0, data)
        distance1 = get_distance(template1, data)
        print('distance0 {} distance1 {}'.format(distance0, distance1))
        if (distance0 < distance1):
            print('right')
            count += 1
        else:
            print('wrong')
    print('count {} / total {} accuracy   {}'.format(count,len(data0),count / len(data0)))
    pass


def get_distance(data0, data1):
    min_distance = 10000
    for data in data0:
        for template in data:
            for i in range(8):
                compared = data1[i]
                distance = norm(template.flatten() - compared.flatten(), ord=2)
                if distance < min_distance:
                    min_distance = distance

    return min_distance


def get_data(dir_path):
    result = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        data = np.load(open(filepath, 'rb'))
        result.append(data)
    result=shuffle(result)
    return result


def shuffle(data_list):
    indexes = np.arange(len(data_list))
    np.random.shuffle(indexes)
    result = []
    for index in indexes:
        result.append(data_list[index])
    return result


if __name__ == '__main__':
    main()
