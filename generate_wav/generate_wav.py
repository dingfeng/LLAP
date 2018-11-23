# -*- coding: UTF-8 -*-
# filename: generate_wav date: 2018/11/23 15:30  
# author: FD 
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftp


def main():
    baseF = 17000
    deltaF = 350
    fn = 16
    fs = 192000
    duration = 5 * 60
    fslist = []
    currentF = baseF + deltaF // 2
    fslist.append(currentF)
    for i in range(fn - 1):
        currentF += deltaF
        fslist.append(currentF)
    data = np.zeros(fs * duration)
    times = np.linspace(0, duration, fs * duration, False)
    for fstmp in fslist:
        fdata = np.cos(2 * np.pi * fstmp * times)
        data += fdata

    wavfile.write('sound.wav', int(fs), data.astype(np.float32))



if __name__ == '__main__':
    main()
    # generate()
