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

name2label = {'gaoshihao': 0, 'huangyuyang': 1, 'kanghuquan': 2, 'linjianghao': 3, 'liujia': 4, 'liuqianxi': 5,
              'liuzhengwei': 6, 'lvzhenyu': 7, 'qipeng': 8, 'shenjiahuan': 9, 'taobocheng': 10, 'wanghaoyu': 11,
              'wangxinzhe': 12, 'wangzhulai': 13, 'xiayuxing': 14, 'xiejiahang': 15, 'yuanjianyong': 16, 'zhaoxing': 17,
              'zhaoxuyang': 18, 'zhuchaoyang': 19, 'zhuwenjie': 20}

label2name = {0: 'gaoshihao', 1: 'huangyuyang', 2: 'kanghuquan', 3: 'linjianghao', 4: 'liujia', 5: 'liuqianxi',
              6: 'liuzhengwei', 7: 'lvzhenyu', 8: 'qipeng', 9: 'shenjiahuan', 10: 'taobocheng', 11: 'wanghaoyu',
              12: 'wangxinzhe', 13: 'wangzhulai', 14: 'xiayuxing', 15: 'xiejiahang', 16: 'yuanjianyong', 17: 'zhaoxing',
              18: 'zhaoxuyang', 19: 'zhuchaoyang', 20: 'zhuwenjie'}

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
    pass


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
