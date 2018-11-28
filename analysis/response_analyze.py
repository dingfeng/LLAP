# -*- coding: UTF-8 -*-
# filename: response_analyze date: 2018/11/25 15:04  
# author: FD 
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
w, h = signal.freqs([3], [1,1,1,1,1,1,1,1])
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()