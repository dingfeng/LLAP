# -*- coding: UTF-8 -*-
# filename: generate_wav date: 2018/11/23 15:30  
# author: FD 
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftp
import matplotlib.pyplot as plt


def main():
    baseF = 17000+175
    deltaF = 700
    fn = 8
    fs = 48000
    duration = 2 * 60
    fslist = []
    currentF = baseF
    fslist.append(currentF)
    for i in range(fn - 1):
        currentF += deltaF
        fslist.append(currentF)
    data = np.zeros(fs * duration)
    times = np.linspace(0, duration, fs * duration, False)
    for fstmp in fslist:
        fdata = np.cos(2 * np.pi * fstmp * times) * np.sqrt(2)
        data += fdata
    data = data / 16
    wavfile.write('sound.wav', int(fs), data.astype(np.float32))


if __name__ == '__main__':
    main()
    # generate()
