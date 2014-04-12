# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:09:00 2014

@author: Jm
"""
import numpy as np
import sklearn.externals.joblib.pool as pol


class NumpyFactory:

    instanceCounter = 0

    def __init__(self, maxBytes=10e6, tmpFolder="tmp/", verbosity=50):
        self._tmpFolder = "tmp/"
        self._maxBytes = 10e6
        #self._id = self._identity()
        self._mmap = pol.ArrayMemmapReducer(max_nbytes=maxBytes,
                                            temp_folder=tmpFolder,
                                            mmap_mode="r+",
                                            verbose=verbosity)

    def _identity(self):
        #TODO Lock
        _id = NumpyFactory.instanceCounter
        NumpyFactory.instanceCounter += 1
        return _id

    def createArray(self, shape):
        size = np.prod(shape)*np.dtype(np.float).itemsize
        if size > self._maxBytes:
            return np.zeros(shape)
        pickler, data = self._mmap(np.zeros(shape))
        return pickler(*data)
