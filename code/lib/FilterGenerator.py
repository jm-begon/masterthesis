# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 22 2014
"""
A set of filter generator and holder.

Filter
------
A filter is 2D numpy array which can be used to filter an image with a
convolution operator
"""
import numpy as np

__all__ = ["FilterGenerator", "FixSizeFilterGenerator", "FiniteFilter",
           "Finite3Filter", "Finite3SameFilter"]


class FilterGenerator:

    """
    ===============
    FilterGenerator
    ===============
    A base class which generate randomly filters with specified features.
    A filter is a 2D numpy array.

    Constants
    ---------
    Normalisation parameters :
    NORMALISATION_NONE : indicate that no normalisation is required.
    NORMALISATION_MEAN : indicate that filter should be normalised to have
    a mean value of 0.
    NORMALISATION_VAR : indicate that filter should be normalised to have
    variance of 1.
    NORMALISATION_MEANVAR : indicate that filter should be normalised to have
    a mean of 0 and a variance of 1.
    """
    NORMALISATION_NONE = 0
    NORMALISATION_MEAN = 1
    NORMALISATION_VAR = 2
    NORMALISATION_MEANVAR = 3
    NORMALISATION_SUMTO1 = 4
    EPSILON = 10**-15

    def __init__(self, numberValueGenerator, numberSizeGenerator,
                 squareSized=True, normalisation=NORMALISATION_NONE):
        """
        Construct a :class:`FilterGenerator`

        Parameters
        ----------
        numberValueGenerator : :class:`NumberGenerator`
            The generator which will draw the values of the filter
        numberSizeGenerator : :class:`NumberGenerator`
            The generator which will draw the size of the filter.
            Must generate positive integers.
        squareSized : boolean (default : True)
            Whether or not the filter should have the same width as height
        normalisation : int (normalisation parameter : {NORMALISATION_NONE,
        NORMALISATION_MEAN, NORMALISATION_VAR, NORMALISATION_MEANVAR}) (
        default : NORMALISATION_NONE)
            The normalisation to apply to the filter
        """
        self._valGen = numberValueGenerator
        self._normalisation = normalisation
        self._sizeGen = numberSizeGenerator
        self._sqr = squareSized

    def _getSize(self):
#        """
#        Generate the height and width
#        """
        height = self._sizeGen.getNumber()
        if self._sqr:
            width = height
        else:
            width = self._sizeGen.getNumber()
        return height, width

    def _normalize(self, filt):
        """
        Normalize the filter according to the instance policy

        Parameters
        ----------
        filt : 2D numpy array
            The filter to normalize

        Return
        ------
        normalizedFilter : 2D numpy array of the same shape
            the normalized filter
        """
        #Normalisation
        if self._normalisation == FilterGenerator.NORMALISATION_SUMTO1:
            return filt/sum(filt)
        if (self._normalisation == FilterGenerator.NORMALISATION_MEAN or
                self._normalisation == FilterGenerator.NORMALISATION_MEANVAR):
            filt = filt - filt.mean()

        if (self._normalisation == FilterGenerator.NORMALISATION_VAR or
                self._normalisation == FilterGenerator.NORMALISATION_MEANVAR):
            stdev = filt.std()
            if abs(stdev) > FilterGenerator.EPSILON:
                filt = filt/stdev
        return filt

    def next(self):
        """
        Return a newly generated filter
        """
        #Generate the size
        height, width = self._getSize()
        #Fill in the values
        linearFilter = np.zeros((height, width))
        for i in xrange(height):
            for j in xrange(width):
                linearFilter[i][j] = self._valGen.getNumber()

        return self._normalize(linearFilter)


class FixSizeFilterGenerator(FilterGenerator):
    """
    ======================
    FixSizeFilterGenerator
    ======================
    Generate filters of constant size
    """
    def __init__(self, numberValueGenerator, height, width,
                 normalisation=FilterGenerator.NORMALISATION_NONE):
        """
        Construct a :class:`FixSizeFilterGenerator`

        Parameters
        ----------
        numberValueGenerator : :class:`NumberGenerator`
            The generator which will draw the values of the filter
        height : int > 0
            The height (number of rows) of the filter
        width : int > 0
            The width (number of columns) of the filter
        squareSized : boolean (default : True)
            Whether or not the filter should have the same width as height
        normalisation : int (normalisation parameter : {NORMALISATION_NONE,
        NORMALISATION_MEAN, NORMALISATION_VAR, NORMALISATION_MEANVAR}) (
        default : NORMALISATION_NONE)
            The normalisation to apply to the filter
        """
        self._valGen = numberValueGenerator
        self._height = height
        self._width = width
        self._normalisation = normalisation

    def _getSize(self):
        return self._height, self._width


class FiniteFilter:
    """
    ============
    FiniteFilter
    ============

    A :class:`FiniteFilter` is a container of filters.
    """

    def __init__(self, filterGenerator, nbFilter):
        """
        Construct a :class:`FiniteFilter`

        Parameters
        ----------
        filterGenerator : :class:`FilterGenerator`
            The generator which will produce the filters
        nbFilter : int > 0
            The number of filters to generate
        """
        filters = []
        for i in xrange(nbFilter):
            filters.append(filterGenerator.next())

        self._filters = filters

    def __iter__(self):
        return iter(self._filters)

    def __len__(self):
        return len(self._filters)


class Finite3Filter(FiniteFilter):
    """
    =============
    Finite3Filter
    =============

    A :class:`Finite3Filter` is a container of filter triplets.
    """
    def __init__(self, filterGenerator, nbFilter):
        """
        Construct a :class:`Finite3Filter`

        Parameters
        ----------
        filterGenerator : :class:`FilterGenerator`
            The generator which will produce the filters
        nbFilter : int > 0
            The number of filter triplets to generate
        """
        filters = []
        for i in xrange(nbFilter):
            filters.append((filterGenerator.next(), filterGenerator.next(),
                            filterGenerator.next()))
        self._filters = filters


class Finite3SameFilter(Finite3Filter):
    """
    =================
    Finite3SameFilter
    =================

    A :class:`Finite3SameFilter` is a container of filter triplets where
    each filter of a triplet are the same
    """

    def __init__(self, filterGenerator, nbFilter):
        """
        Construct a :class:`Finite3SameFilter`

        Parameters
        ----------
        filterGenerator : :class:`FilterGenerator`
            The generator which will produce the filters
        nbFilter : int > 0
            The number of filter triplets to generate
        """
        filters = []
        for i in xrange(nbFilter):
            filt = filterGenerator.next()
            filters.append((filt, filt, filt))

        self._filters = filters


if __name__ == "__main__":
    test = True
    if test:
        from NumberGenerator import (NumberGenerator,
                                     IntegerUniformGenerator,
                                     OddUniformGenerator)

        fltGen = FilterGenerator(NumberGenerator(-7, 15),
                                 OddUniformGenerator(30, 16))

        fltGen2 = FilterGenerator(IntegerUniformGenerator(15, 20),
                                  OddUniformGenerator(3, 10))

        diff3 = Finite3Filter(fltGen, 5)

        same3 = Finite3SameFilter(fltGen2, 5)

        for f in diff3:
            print f[0], f[0].shape
            print f[1], f[1].shape
            print f[2], f[2].shape

        for f in same3:
            print f[0], f[0].shape
            print f[1], f[1].shape
            print f[2], f[2].shape
