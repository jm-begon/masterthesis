# -*- coding: utf-8 -*-
"""
Created on Sat Aug 02 15:03:12 2014

@author: Jm
"""
import numpy as np
import itertools
import random


class ImageSuffler:

    def __init__(self, height, width, rate):
        if rate > 0:
            coords = itertools.product(range(height), range(width))
            coordsList = [x for x in coords]
            random.shuffle(coordsList)
            self._transfert = coordsList[0:int(height*width*rate)]

    def shuffle(self, numpImg):
        result = np.copy(numpImg)
        if not hasattr(self, "_transfert"):
            return result

        previous = self._transfert[-1]
        current = self._transfert[0]
        for i in xrange(len(self._transfert)):
            p_r, p_c = previous
            c_r, c_c = current
            result[p_r, p_c] = numpImg[c_r, c_c]
            previous = current
            current = self._transfert[i % (len(self._transfert))]

        return result


def imgNormDist(numpOri, numpRes):
    d = sum(abs(numpOri - numpRes))
    height, width = numpOri.shape[0], numpOri.shape[1]
    return d / float(height*width)
