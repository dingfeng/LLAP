# -*- coding: UTF-8 -*-
# filename: GenerateDataset date: 2019/4/6 15:20  
# author: FD 
import numpy as np
import os
from PIL import Image, ImageFilter
import pickle

name2label = {'anna': 0, 'chenbo': 1, 'chenhao': 2, 'dengyufeng': 3, 'dingfeng': 4, 'huangsi': 5, 'jianghao': 6,
              'qingpeijie': 7,
              'xuhuatao': 8, 'yinjunhao': 9, 'yuyinggang': 10, 'zhangqian': 11, 'zhaorun': 12, 'zhuyan': 13}

label2name = {0: 'anna', 1: 'chenbo', 2: 'chenhao', 3: 'dengyufeng', 4: 'dingfeng', 5: 'huangsi', 6: 'jianghao',
              7: 'qingpeijie', 8: 'xuhuatao', 9: 'yinjunhao', 10: 'yuyinggang', 11: 'zhangqian', 12: 'zhaorun',
              13: 'zhuyan'}

genuine_num = 100
skilled_forged_num = 50
random_forged_num = 50

result_x = []
result_y = []
result_yforg = []


def main():
    global genuine_num
    global skilled_forged_num
    global random_forged_num
    global result_x
    global result_x
    global result_yforg
    names = list(name2label.keys())
    np.random.shuffle(names)
    name_len = len(names)
    for i in range(name_len):
        name = names[i]
        # genuine signature
        name_path = os.path.join('./data/cutted', name)
        filenames = os.listdir(name_path)
        np.random.shuffle(filenames)
        filenames = filenames[:genuine_num]
        for filename in filenames:
            filepath = os.path.join(name_path, filename)
            image_array = get_array(filepath)
            x = image_array.reshape((1, image_array.shape[0], image_array.shape[1]))
            y = name2label[name]
            yforg = 0
            add2result(x, y, yforg)
        # forged signature
        for j in range(1, 6):
            index = j % len(names)
            forged_name = names[index]
            forged_name_path = os.path.join('./data/cutted', forged_name)
            forged_name_filenames = os.listdir(forged_name_path)
            np.random.shuffle(forged_name_filenames)
            forged_name_filenames = forged_name_filenames[:10]
            for forged_filename in forged_name_filenames:
                forged_filepath = os.path.join(forged_name_path, forged_filename)
                image_array = get_array(forged_filepath)
                x = image_array.reshape((1, image_array.shape[0], image_array.shape[1]))
                y = name2label[name]
                yforg = 1
                add2result(x, y, yforg)
        # skill signature
        skill_forged_name_path = os.path.join('./data/forged_cutted',name)
        skill_forged_filenames = os.listdir(skill_forged_name_path)
        skill_forged_filenames = skill_forged_filenames[:skilled_forged_num]
        for skill_forged_filename in skill_forged_filenames:
            skill_forged_filepath = os.path.join(skill_forged_name_path, skill_forged_filename)
            image_array = get_array(skill_forged_filepath)
            x = image_array.reshape((1, image_array.shape[0], image_array.shape[1]))
            y = name2label[name]
            yforg = 1
            add2result(x, y, yforg)
    todump_obj = {'x': np.asarray(result_x), 'y': np.asarray(result_y), 'yforg': np.asarray(result_yforg),'user_mapping':[1],'filenames':[1]}
    pickle.dump(todump_obj,open('./data/dataset/sig_dataset.npz','wb'))


def add2result(x, y, yforg):
    global result_x
    global result_y
    global result_yforg
    result_x.append(x)
    result_y.append(y)
    result_yforg.append(yforg)


def get_array(image_path):
    im = Image.open(image_path).convert('L')
    array = np.asarray(im)
    return array


if __name__ == '__main__':
    main()
