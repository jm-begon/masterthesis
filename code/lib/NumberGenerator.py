# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 22 2014
"""
A set of number generator
"""
from sklearn.utils import check_random_state

__all__ = ["NumberGenerator", "IntegerUniformGenerator", "OddUniformGenerator"]


class NumberGenerator:
    """
    ===============
    NumberGenerator
    ===============
    A random number generator which draws number between two bounds :
    [min, max)
    This base class returns uniform real number
    """
    def __init__(self, minVal=0, maxVal=1, seed=None):
        """
        Construct a :class:`NumberGenerator`

        Parameters
        ----------
        minVal : float (default : 0)
            The minimum value from which to draw
        maxVal : float (default : 1)
            The maximum value from which to draw
        seed : int or None (default : None)
            if seed is int : initiate the random generator with this seed
        """
        self._randGen = check_random_state(seed)
        self._min = minVal
        self._max = maxVal

    def _doGetNumber(self, minVal, maxVal):
#        """
#        Return a random number comprise between [minVal, maxVal)
#
#        Parameters
#        ----------
#        minVal : float
#            The minimum value from which to draw
#        maxVal : float
#            The maximum value from which to draw
#
#        Return
#        ------
#        rand : float
#            A random number comprise between [minVal, maxVal)
#        """
        return minVal+self._randGen.rand()*(maxVal-minVal)

    def getNumber(self, minVal=None,  maxVal=None):
        """
        Return a random number comprise between [minVal, maxVal)

        Parameters
        ----------
        minVal : float/None (default : None)
            The minimum value from which to draw
            if None : use the class minimum
        maxVal : float/None (default : None)
            The maximum value from which to draw
            if None : use the class maximum

        Return
        ------
        rand : float
            A random number comprise between [minVal, maxVal)
        """
        if minVal is None:
            minVal = self._min
        if maxVal is None:
            maxVal = self._max
        return self._doGetNumber(minVal, maxVal)


class IntegerUniformGenerator(NumberGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform integer number
    """
    def _doGetNumber(self, minVal, maxVal):
        return self._randGen.randint(minVal, maxVal)


class OddUniformGenerator(IntegerUniformGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform odd integer number
    """
    def _doGetNumber(self, minVal, maxVal):
        if maxVal % 2 == 0:
            maxVal -= 1
        if minVal % 2 == 0:
            minVal += 1
        return minVal + 2*int(self._randGen.rand()*((maxVal - minVal)/2+1))
