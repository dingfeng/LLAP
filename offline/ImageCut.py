# -*- coding: UTF-8 -*-
# filename: ImageCut date: 2019/4/2 20:42  
# author: FD 
from PIL import Image, ImageFilter
import numpy as np
from scipy.linalg import norm
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy import ndimage

# len = 485.0505128334574  # norm(np.asarray([511, 370] - np.asarray([996, 363])), 2)
# width = 312.0256399721023  # norm(np.asarray([537, 368] - np.asarray([541, 680])), 2)
len_step = 728.0830997
width_step = 310.0258054
len = 483
width = 312
move_distance = 273
X_vector = np.asarray([332, 1058, 1792, 2522]) - 332
Y_vector = np.asarray([395, 707, 1038, 1353, 1684, 2030, 2372]) - 332


def main():
    rotate()


def cut():
    print('cut')


def im2array(im):
    gimg_ndarr = np.asarray(im, dtype='float64')
    return gimg_ndarr


def rotate():
    rootDir = './data'
    rawDir = os.path.join(rootDir, 'forged_raw')
    cuttedDir = os.path.join(rootDir, 'forged_cutted')
    personDirs = os.listdir(rawDir)
    for personDir in personDirs:
        if personDir!='test1':
            continue
        personDirPath = os.path.join(rawDir, personDir)
        destPersonDirPath = os.path.join(cuttedDir, personDir)
        if not os.path.isdir(destPersonDirPath):
            os.mkdir(destPersonDirPath)
        confFilepath = os.path.join(personDirPath, 'conf.txt')
        confs = np.loadtxt(confFilepath)
        filenames = os.listdir(personDirPath)
        filenames = np.sort(filenames).tolist()
        for i in range(4):
            filename = ''.join([filenames[i]])
            if filename.endswith('txt'):
                continue
            filepath = os.path.join(personDirPath, filename)
            # prefix = filename[:-4]
            im = Image.open(filepath)
            conf = confs[i, :]
            angle = get_angle(conf)
            rotated_im = im.rotate(-angle, expand=True, center=(conf[0], conf[1]))
            for m in range(4):
                for n in range(7):
                    x_pos = conf[0] + X_vector[m]
                    y_pos = conf[1] + Y_vector[n]
                    y_pos -= move_distance
                    cropped_im = rotated_im.crop([x_pos, y_pos, x_pos + len, y_pos + width])
                    dest_filepath = os.path.join(destPersonDirPath,
                                                 ''.join([str(i + 1), '-', str(m + 1), '-', str(n + 1), '.jpg']))
                    cropped_im.save(dest_filepath)
                    # cropped_im.show()

                    # rotated_im.save(dest_filepath)
    return


def get_angle(conf):
    return np.arctan((conf[1] - conf[3]) / (conf[2] - conf[0])) / (0.5 * np.pi) * 90


def process_image(filepath, prefix):
    im = Image.open(filepath)
    img_size = im.size
    print('image size {}'.format(img_size))


if __name__ == '__main__':
    main()
