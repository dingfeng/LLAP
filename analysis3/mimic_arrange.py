# -*- coding: UTF-8 -*-
# filename: mimic_arrange date: 2019/1/13 14:22  
# author: FD 
import os
from shutil import copyfile


def main():
    # mimic_arrange()
    mimic_arrange()

def mimic_arrange():
    root_dir = '../dataset/handwriting-lab-3/mimic_cutted/'
    dest_dir = '../dataset/handwriting-lab-3/mimic_cutted_arranged'
    dir_names = os.listdir(root_dir)
    for dir_name in dir_names:
        dir_path = os.path.join(root_dir, dir_name)
        mimic_dir_names = os.listdir(dir_path)
        for mimic_dir_name in mimic_dir_names:
            mimic_dir_name_path = os.path.join(dir_path, mimic_dir_name)
            mimic_filenames = os.listdir(mimic_dir_name_path)
            for filename in mimic_filenames:
                filepath = os.path.join(mimic_dir_name_path, filename)
                dest_dir_path = os.path.join(dest_dir, mimic_dir_name)
                if not os.path.isdir(dest_dir_path):
                    os.makedirs(dest_dir_path)
                dest_filepath = os.path.join(dest_dir_path, ''.join([dir_name, '-', filename]))
                copyfile(filepath, dest_filepath)

def mimic_total():
    root_dir = '../dataset/handwriting-lab-1/mimic-cutted2/'
    dest_dir = '../dataset/handwriting-lab-1/mimic_cutted_total2'
    dir_names = os.listdir(root_dir)
    for dir_name in dir_names:
        dir_path = os.path.join(root_dir, dir_name)
        dest_dir_path = os.path.join(dest_dir, dir_name)
        mimic_dir_names = os.listdir(dir_path)
        for mimic_dir_name in mimic_dir_names:
            mimic_dir_name_path = os.path.join(dir_path, mimic_dir_name)
            mimic_filenames = os.listdir(mimic_dir_name_path)
            for filename in mimic_filenames:
                filepath = os.path.join(mimic_dir_name_path, filename)
                if not os.path.isdir(dest_dir_path):
                    os.makedirs(dest_dir_path)
                dest_filepath = os.path.join(dest_dir_path, ''.join([mimic_dir_name, '-', filename]))
                copyfile(filepath, dest_filepath)

if __name__ == '__main__':
    main()
