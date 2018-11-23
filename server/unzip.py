# -*- coding: UTF-8 -*-
# filename: unzip date: 2018/11/22 10:46  
# author: FD
import os
from zipfile import ZipFile as zip


def extractAll(zipName):
    dirname = zipName[0:zipName.index('.zip')]
    if (os.path.exists(dirname)):
        return
    os.makedirs(dirname)
    z = zip(zipName)
    os.chdir(dirname)
    for f in z.namelist():
        if f.endswith('/'):
            os.makedirs(f)
        else:
            z.extract(f)
    os.chdir('../')


if __name__ == '__main__':
    for filename in os.listdir("."):
        if filename.endswith(".zip"):
            extractAll(filename)
