# -*- coding: UTF-8 -*-
# filename: filter_response date: 2018/12/27 15:48  
# author: FD 
import numpy as np

from pylab import *
import scipy.signal as signal
def mfreqz(b,a=1):
    w,h = signal.freqz(b,a)
    h = abs(h)
    return(w/max(w), h)

n = 100.
b = repeat(1/n, n)
w, h = mfreqz(b)
#Plot the function
for i in range(len(h)):
    if h[i]<0.002:
        print(w[i])
plot(w,h,'k')
ylabel('Amplitude')
xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
title(r'Frequency response of an 11 point moving average filter')
# savefig('ma_freq.pdf')
show()