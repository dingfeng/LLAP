# -*- coding: UTF-8 -*-
# filename: CIC date: 2018/11/23 20:44  
# author: FD 
import numpy as np


class CIC:
    def __init__(self, decimationFactor, numberOfSections, diferrencialDelay):
        self.R = decimationFactor
        self.N = numberOfSections
        self.M = diferrencialDelay
        self.buffer_comb = None
        self.buffer_integrator = None
        self.offset_comb = None
        assert self.R > 0 and self.N > 0 and self.M > 0
        self.buffer_integrator = np.zeros(self.N)
        self.offset_comb = np.zeros(self.N).astype(np.int)
        self.buffer_comb = np.zeros((self.N, self.M))
        return

    def reset(self):
        self.buffer_integrator.fill(0)
        self.offset_comb.fill(0)
        self.buffer_comb.fill(0)
        return

    def filter(self, data):
        if len(data) != self.R:
            return 0
        anttenuation = 1.0
        tmp_out = 0
        for i in range(self.R):
            tmp_out = data[i]
            for j in range(self.N):
                tmp_out = self.buffer_integrator[j] = self.buffer_integrator[j] + tmp_out

        for i in range(self.N):
            self.offset_comb[i] = (self.offset_comb[i] + 1) % self.M
            tmp = self.buffer_comb[i][self.offset_comb[i]]
            self.buffer_comb[i][self.offset_comb[i]] = tmp_out
            tmp_out = tmp_out - tmp

        return anttenuation * tmp_out
