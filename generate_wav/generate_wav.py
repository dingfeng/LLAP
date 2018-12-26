# -*- coding: UTF-8 -*-
# filename: generate_wav date: 2018/11/23 15:30  
# author: FD 
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftp
import matplotlib.pyplot as plt


def main():
    baseF = 17350
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
    weights = np.array([1996.57178289585, 1130.8621028539283, 695.1034288151462, 663.849980505101, 533.3391836417595,
                        628.0630697248464, 896.4878967366324,1112.6761164155364])
    # weights=np.ones((1,fn)).flatten()
    weights = 1 / weights
    for index, fstmp in enumerate(fslist):
        fdata = weights[index] * np.cos(2 * np.pi * fstmp * times)  # * np.sqrt(2)
        data += fdata
    data = data / np.sum(weights)
    # xf = np.arange(len(data)) / len(data) * 48000
    # yf = fftp.fft(data, len(data))
    # yf = np.abs(yf)
    # # for i in range(20):
    # #     yf[np.argmax(yf)] = 0
    # plt.plot(xf, yf)
    # plt.show()
    data = data * 32767
    wavfile.write('sound.wav', int(fs), data.astype(np.int16))


if __name__ == '__main__':
    main()
    # generate()
