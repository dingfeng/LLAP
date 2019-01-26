# -*- coding: UTF-8 -*-
# filename: plot_fft date: 2019/1/26 20:00  
# author: FD 
import scipy.fftpack as fftp
import numpy as np
import matplotlib.pyplot as plt
def main():
    source_filepath='multi-freq-nocompensation.pcm'
    data = np.memmap(source_filepath, dtype=np.float32, mode='r')
    plot_fft(data[48000*3:48000*4])

def plot_fft(data,fs=48000):
    plt.figure(figsize=(10,5))
    xf = np.arange(len(data)) / len(data) * fs
    yf = fftp.fft(data, len(data))
    yf = np.abs(yf)
    plt.xlabel('Frequency (Hz)',fontdict={'style': 'normal', 'weight': 'bold','size':22})
    plt.ylabel('Magnitude',fontdict={'style': 'normal', 'weight': 'bold','size':22})
    plt.xticks(fontsize=17,fontname='normal')
    plt.yticks(fontsize=17,fontname='normal')
    plt.plot(xf, yf)
    plt.tight_layout()
    plt.savefig('nocompensation.pdf', dpi=100)
    plt.show()

if __name__ == '__main__':
    main()