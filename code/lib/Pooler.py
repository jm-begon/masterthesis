# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 03 2014
"""
A set of classes which realise spatial pooling
"""
from abc import ABCMeta, abstractmethod

__all__ = ["Pooler", "MultiPooler", "IdentityPooler", "ConvolutionalPooler",
           "ConvMaxPooler", "ConvMinPooler", "ConvAvgPooler"]


class Pooler:
    """
    ======
    Pooler
    ======
    A :class:`Pooler` performs spatial pooling of a 2D or 3D numpy array.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def pool(self, npArray):
        """
        Aggregate the `npArray`

        Parameters
        -----------
        npArray : 2D or 3D numpy array
            The array to aggregate

        Return
        -------
        aggregated : 2D or 3D numpy array (depending on `npArray`)
            The aggregated array
        """
        pass

    def __call__(self, npArray):
        """
        Delegate to meth:`pool` method
        """
        return self.pool(npArray)


class MultiPooler:
    """
    ===========
    MultiPooler
    ===========
    A :class:`MultiPooler` performs several pooling and returns an iterable
    of results
    """
    def __init__(self, poolerList=[]):
        """
        Construct a :class:`MultiPooler` instance

        Parameters
        ----------
        poolerList : iterable of :class:`Pooler` (default : [])
            The individual :class:`Pooler`s
        """
        self._pooler = poolerList

    def multipool(self, npArray):
        """
        Apply the :class:`Pooler`s to the array

        Parameters
        -----------
        npArray : 2D or 3D numpy array
            The array to aggregate

        Return
        -------
        ls : an iterable of 2D or 3D numpy array (depending on `npArray`)
            The results of the individual :class:`Pooler`s (in the same order)
        """
        if len(self._pooler) == 0:
            return [npArray]
        ls = []
        for pooler in self._pooler:
            ls.append(pooler.pool(npArray))

        return ls


class IdentityPooler(Pooler):
    """
    ==============
    IdentityPooler
    ==============
    A :class:`IdentityPooler` return the array untouched (no spatial pooling)
    """

    def pool(self, npArray):
        return npArray


class ConvolutionalPooler(Pooler):
    """
    ===================
    ConvolutionalPooler
    ===================
    A class:`ConvolutionalPooler` works by using a convolutional window in
    which it uses a possibly non-linear function. Unless mentionned otherwise,
    in the case of 3D, the class:`ConvolutionalPooler` works separately on
    each 2D subarray (along the 3rd dimension) and then concatenate the
    results in the 3rd dimension order.
    """

    def __init__(self, function, height, width):
        self._function = function
        self._height = height
        self._width = width

    def pool(self, npArray):
        pass  # TODO XXX


class ConvMaxPooler(ConvolutionalPooler):
    def __init__(self, height, width):
        ConvolutionalPooler.__init__(self, self._max, height, width)

    def _max(self, npArray):
        return npArray.max()


class ConvMinPooler(ConvolutionalPooler):
    def __init__(self, height, width):
        ConvolutionalPooler.__init__(self, self._min, height, width)

    def _min(self, npArray):
        return npArray.min()


class ConvAvgPooler(ConvolutionalPooler):
    def __init__(self, height, width):
        ConvolutionalPooler.__init__(self, self._mean, height, width)

    def _mean(self, npArray):
        return npArray.mean()
