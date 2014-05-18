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

class FastMWPooler(Pooler):
    def __init__(self, fastFunction, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        self._function = fastFunction
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def pool(self, npArray):
		if npArray.ndim == 2:
			return self._function(npArray, self._windowHalfHeight, self._windowHalfWidth)
		ls = []
          cdef unsigned int i
		for i in xrange(npArray.shape[2]):
			ls.append(self._function(npArray[:,:,i], self._windowHalfHeight, self._windowHalfWidth))
		return np.dstack(ls)


class FastMWAvgPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        FastMWPooler.__init__(avgPooling, height, width)

class FastMWMaxPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
		FastMWPooler.__init__(maxPooling, height, width)


class FastMWMinPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        FastMWPooler.__init__(minPooling, height, width)




@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def avgPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol, counter
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double acc
    height = img.shape[0]
    width = img.shape[1]

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
                    acc = acc + img[subrow, subcol]
                    counter = counter + 1

            result[row, col] = acc/counter

    return result

@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def maxPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double maxVal
    height = img.shape[0]
    width = img.shape[1]

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

            maxVal = img[u, l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow, subcol] > maxVal:
                        maxVal = img[subrow, subcol]

            result[row, col] = maxVal

    return result


@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def minPooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double minVal
    height = img.shape[0]
    width = img.shape[1]

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

            minVal = img[u, l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow, subcol] < minVal:
                        minVal = img[subrow, subcol]

            result[row, col] = minVal

    return result