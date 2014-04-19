# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:57:26 2014

@author: Jm
"""
import numpy as np


class Rescaler:

    def __init__(self):
        pass

    def rescale(self, val):
        return val

    def __call__(self, val):
        return self.rescale(val)


class MaxoutRescaler(Rescaler):

    def __init__(self, dtype=np.uint8):
        try:
            info = np.iinfo(dtype)
        except:
            info = np.finfo(dtype)
        self._min = info.min
        self._max = info.max
        self._buffer = np.zeros((1), dtype)

    def rescale(self, val):
        if val < self._min:
            return self._min
        if val > self._max:
            return self._max
        self._buffer[0] = val
        return self._buffer[0]
