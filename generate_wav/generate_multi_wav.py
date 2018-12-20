# -*- coding: UTF-8 -*-
# filename: generate_wav date: 2018/11/23 15:30
# author: FD
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftp
import matplotlib.pyplot as plt


def main():
    baseF = 17700
    deltaF = 700
    fn = 8
    fs = 48000
    duration = 2 * 60
    fclist = []
    currentF = baseF
    fclist.append(currentF)
    for i in range(fn - 1):
        currentF += deltaF
        fclist.append(currentF)
    data = np.zeros(fs * duration)
    times = np.linspace(0, duration, fs * duration, False)
    for fstmp in fclist:
        fdata = np.cos(2 * np.pi * fstmp * times)  # * np.sqrt(2)
        data += fdata
    data = data / fn * 32767
    wavfile.write('sound.wav', int(fs), data.astype(np.int16))


if __name__ == '__main__':
    main()
    # generate()
