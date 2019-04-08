# -*- coding: UTF-8 -*-
# filename: GenerateDataset date: 2019/4/6 15:20
# author: FD
import numpy as np
import os
from PIL import Image, ImageFilter
import pickle
from skimage.io import imread
from skimage import img_as_ubyte
from sigver.preprocessing.normalize import preprocess_signature

name2label = {'gaoshihao': 1000, 'huangyuyang': 1001, 'kanghuquan': 1002, 'linjianghao': 1003, 'liujia': 1004, 'liuqianxi': 1005,
              'liuzhengwei': 1006, 'lvzhenyu': 1007, 'qipeng': 1008, 'shenjiahuan': 1009, 'taobocheng': 1010, 'wanghaoyu': 1011,
              'wangxinzhe': 1012, 'wangzhulai': 1013, 'xiayuxing': 1014, 'xiejiahang': 1015, 'yuanjianyong': 1016, 'zhaoxing': 1017,
              'zhaoxuyang': 1018, 'zhuchaoyang': 1019, 'zhuwenjie': 1020}

label2name = {1000: 'gaoshihao', 1001: 'huangyuyang', 1002: 'kanghuquan', 1003: 'linjianghao', 1004: 'liujia', 1005: 'liuqianxi',
              1006: 'liuzhengwei', 1007: 'lvzhenyu', 1008: 'qipeng', 1009: 'shenjiahuan', 1010: 'taobocheng', 1011: 'wanghaoyu',
              1012: 'wangxinzhe', 1013: 'wangzhulai', 1014: 'xiayuxing', 1015: 'xiejiahang', 1016: 'yuanjianyong', 1017: 'zhaoxing',
              1018: 'zhaoxuyang', 1019: 'zhuchaoyang', 1020: 'zhuwenjie'}

genuine_num = 32
skilled_forged_num = 15

result_x = []
result_y = []
result_yforg = []


def main():
    global genuine_num
    global skilled_forged_num
    global result_x
    global result_y
    global result_yforg
    names = list(name2label.keys())
    np.random.shuffle(names)
    name_len = len(names)
    for i in range(name_len):
        name = names[i]
        # genuine signature
        name_path = os.path.join('./testdata/cutted', name)
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
        # skill signature
        skill_forged_name_path = os.path.join('./testdata/forged', name)
        skill_forged_filenames = os.listdir(skill_forged_name_path)
        skill_forged_filenames = skill_forged_filenames[:skilled_forged_num]
        for skill_forged_filename in skill_forged_filenames:
            skill_forged_filepath = os.path.join(skill_forged_name_path, skill_forged_filename)
            print('path {}'.format(skill_forged_filepath))
            image_array = get_array(skill_forged_filepath)
            x = image_array.reshape((1, image_array.shape[0], image_array.shape[1]))
            y = name2label[name]
            yforg = 1
            add2result(x, y, yforg)
    todump_obj = {'x': np.asarray(result_x), 'y': np.asarray(result_y), 'yforg': np.asarray(result_yforg),
                  'user_mapping': [1], 'filenames': [1]}
    pickle.dump(todump_obj, open('./testdata/dataset/sig_exp_dataset.npz', 'wb'))
    fuse_datasets()


def fuse_datasets():
    dev_dataset = np.load(open('./data/dataset/sig_dev_dataset.npz', 'rb'))
    exp_dataset = np.load(open('./testdata/dataset/sig_exp_dataset.npz', 'rb'))
    exp_dataset['x'] = np.vstack((dev_dataset['x'], exp_dataset['x']))
    exp_dataset['y'] = np.hstack((dev_dataset['y'], exp_dataset['y']))
    exp_dataset['yforg']=np.hstack((dev_dataset['yforg'],exp_dataset['yforg']))
    pickle.dump(exp_dataset, open('./testdata/dataset/sig_exp_dataset_fused.npz', 'wb'))



def add2result(x, y, yforg):
    global result_x
    global result_y
    global result_yforg
    result_x.append(x)
    result_y.append(y)
    result_yforg.append(yforg)


def get_array(image_path):
    canvas_size = (500, 500)
    original = img_as_ubyte(imread(image_path, as_gray=True))
    processed = preprocess_signature(original, canvas_size)
    return processed


if __name__ == '__main__':
    main()
