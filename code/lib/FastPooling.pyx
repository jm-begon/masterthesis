# -*- coding: utf-8 -*-
"""
Created on Sun May 18 08:55:41 2014

@author: Jm Begon
"""
import numpy as np
cimport numpy as np
cimport cython

from Pooler import Pooler

__all__ = ["FastMWAvgPooler", "FastMWMaxPooler", "FastMWMinPooler"]

class FastMWAvgPooler(Pooler):

    def __init__(self, function, height, width):
        """
        Construc a class:`ConvolutionalPooler` instance

        Parameters
        ----------
        function : callable
            A function as mentionned in the class description
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        self._function = function
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def pool(self, npArray):
        return avgPooling(npArray, self._windowHalfHeight, _windowHalfWidth)

class FastMWMaxPooler(Pooler):

    def __init__(self, function, height, width):
        """
        Construc a class:`ConvolutionalPooler` instance

        Parameters
        ----------
        function : callable
            A function as mentionned in the class description
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        self._function = function
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def pool(self, npArray):
        return maxPooling(npArray, self._windowHalfHeight, _windowHalfWidth)

class FastMWMinPooler(Pooler):

    def __init__(self, function, height, width):
        """
        Construc a class:`ConvolutionalPooler` instance

        Parameters
        ----------
        function : callable
            A function as mentionned in the class description
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        self._function = function
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def pool(self, npArray):
        return minPooling(npArray, self._windowHalfHeight, _windowHalfWidth)


@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def avgPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol, counter
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double acc
    height = npArray.shape[0]
    width = npArray.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            counter = 0
            acc = 0.

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    acc = acc + img[subrow][subcol]
                    counter = counter + 1

            result[row][col] = acc/counter

    return result

@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def maxPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double maxVal
    height = npArray.shape[0]
    width = npArray.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            maxVal = img[u][l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow][subcol] > maxVal:
                        maxVal = img[subrow][subcol]

            result[row][col] = maxVal

    return result


@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def minPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double minVal
    height = npArray.shape[0]
    width = npArray.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            minVal = img[u][l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow][subcol] < minVal:
                        minVal = img[subrow][subcol]

            result[row][col] = minVal

    return result