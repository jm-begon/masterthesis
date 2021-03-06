# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of feature extractors
"""
from abc import ABCMeta, abstractmethod

import numpy as np

__all__ = ["Transformer", "Extractor", "IdentityExtractor", "Combinator",
           "StatelessExtractor", "ImageLinearizationExtractor",
           "DepthCompressorILE"]


class Transformer:
    """
    =========
    Transformer
    =========
    :class:`Transformer` is a class responsible for processing some data
    A :class:`Transformer` must specify what it takes as input and how it
    transform it as output
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def transformWholeData(self, wholeData):
        """
        Process the whole data

        Parameters
        ----------
        wholeData : object(s)
            The whole data

        Return
        ------
        transformedData : object(s)
            The corresponding objects
        """
        pass

    @abstractmethod
    def TransformPieceOfData(self, pieceOfData):
        """
        Process a piece of data

        Parameters
        ----------
        pieceOfData : object
            The piece of data

        Return
        ------
        transformedData : object
            The corresponding data
        """
        pass


class Extractor(Transformer):
    """
     =========
     Extractor
     =========
     An :class:`Extractor `is a :class:`Transformer` such that
     - The input verifies :
         wholeData : An iterable of pieceOfData
     - The output verifies :
         - for a piece of data :
             a 1D numpy array for each piece of data (the feature vector)
         - for the whole data :
             a 2D numpy array. Each row of the 2D array is the feature vector
             of the corresponding piece of data

     This base class transform a list of feature vectors into a 2D numpy
     matrix
     """

    def extract(self, obj):
        """
        The :meth:`extract` method implements the feature extraction
        mechanism for one specific piece of data.

        In this base class case, it simply returns the piece of data untouched

        Parameters
        ----------
        obj : extractor-specific-data
            some data

        Return
        ------
        obj itself
        """
        return obj

    def TransformPieceOfData(self, pieceOfData):
        return self.extract(pieceOfData)

    def transformWholeData(self, wholeData):
        return self.transform(wholeData)

    def transform(self, X):
        """
        Apply :meth:`extract` row by row

        Parameter
        ---------
        X : Iterable of extractor-specific-data

        Return
        ------
        npX : numpy array
            The numpy arrayization of the row by row extracted data
        """
        newRow = []
        for i in range(len(X)):
            newRow.append(self.extract(X[i]))
        return np.array(newRow)

    def __call__(self, X):
        """Delegates to :meth:`transform`"""
        return self.transform(X)


class IdentityExtractor(Extractor):
    """
    =================
    IdentityExtractor
    =================

    Returns the data untouched. Therefore, there is no restriction on data.
    """

    def transform(self, X):
        """Returns directly X"""
        return X


class Combinator(Extractor):
    """
    ==========
    Combinator
    ==========

    Combine severals :class:`Extractor`s by concataning horizontally
    their results, in the same order as listed.

    Data
    ----
    The data must compatible with the combination of :class:`Extractor`
    """

    def __init__(self, combination):
        """
        Construct a :class:`Combinator`

        Parameters
        ----------
        combination : Iterable of Extractors
        """
        self._combination = combination

    def extract(self, row):
        newRow = self._combination[0].extract(row)
        for i in range(1, len(self._combination)):
            newRow = np.hstack((newRow, self._combination[i].extract(row)))
        return newRow


class StatelessExtractor(Extractor):
    """
    ==================
    StatelessExtractor
    ==================

    Delegates the :meth:`extract` to a function

    Data
    ----
    The data must compatible with the function
    """

    def __init__(self, function):
        """
        Construct a :class:`StatelessExtractor`

        Parameters
        ----------
        function : callable
            The function to apply on each piece of data
        """
        self._f = function

    def extract(self, row):
        return self._f(row)


class NumpyTransformer(Extractor):
    """
    ================
    NumpyTransformer
    ================
    An :class:`ImageTransformer` instance takes as input :
    - wholeData : iterable of pieceOfData
    - pieceOfData : a 2D or 3D numpy array
    """
    __metaclass__ = ABCMeta
    pass


class ImageLinearizationExtractor(NumpyTransformer, Extractor):
    """
    ===========================
    ImageLinearizationExtractor
    ===========================

    Creates a feature vector of an image by appending each row one after
    the other.

    Data
    ----
    Can deal with numpy 2D (grey image) or 3D (RGB image) array. There might
    be more than 3 colorbands, actually.
    """

    def _lin(self, band):
#        """
#        Linearize one band (2D numpy array)

#        Parameters
#        ----------
#        band : 2D numpy array
#            the band to linearize
#
#        Return
#        ------
#        reshaped : 1D numpy array
#            the array obtained by appending each row one after the other.
#        """
        height, width = band.shape[0], band.shape[1]
        return band.reshape(width*height)

    def extract(self, img):
        """
        Linearize an image ( numpy 2D (grey image) or 3D (RGB image) array)
        by by appending each row one after the other.

        Parameters
        ----------
        img : 2D numpy array or 3D numpy array
            The image to linearize

        Return
        ------
        reshaped : 1D numpy array
            - 2D case
                The array obtained by appending each row one after the other.
            - 3D case
                Similar to the 2D case where each pixel is treated separatedly
                first.

        Example
        -------
        >>> img = np.arange(27).reshape(3,3,3)
        >>> img
            array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],

           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],

           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])
        >>> extractor = ImageLinearizationExtractor()
        >>> extractor.extract(img)
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        """
        shape = img.shape
        size = 1
        for dim in shape:
            size *= dim
        return img.reshape(size)
#
#        if len(img.shape) == 2:
#            #Grey level img
#            return self._lin(img)
#        else:
#            #RGB
#            lin = []
#            for depth in range(img.shape[2]):
#                lin.append(self._lin(img[:, :, depth]))
#            return np.hstack(lin)

    def nbFeaturesPerObject(self, height, width, depth):
        """
        Return the number of features that this :class:`Coordinator` will
        produce per object

        Parameters
        ----------
        height : int > 0
            The height of the image
        width : int > 0
            The width of the image
        depth : int >= 1
            The number of colors
        """
        return height*width*depth


class DepthCompressorILE(ImageLinearizationExtractor):

    def __init__(self, nbColor=2):
        self._nbCol = nbColor
        self._resLenght = 0
        self._depth = 0

    def extract(self, img):
        res = ImageLinearizationExtractor.extract(self, img)
        if len(img.shape) < 3 or img.shape[2] < self._nbCol:
            return res
        lenRes = len(res)
        depth = img.shape[2]
        if self._resLenght != lenRes or self._depth != depth:
            self._resLenght = lenRes
            self._depth = depth
            self._indices = []
            j = 0
            for i in xrange(lenRes):
                if (i + j) % depth < self._nbCol:
                    self._indices.append(i)
                if (i+1) % depth == 0:
                    j += 1

        return res[self._indices]

    def nbFeaturesPerObject(self, height, width, depth):
        if depth == 1:
            return height*width
        if depth < self._nbCol:
            raise ValueError("Not enough colors")
        return height*width*self._nbCol

if __name__ == "__main__":
    test = False
    if test:
        imgpath = "lena.png"
        try:
            import Image
        except:
            from PIL import Image
        img = np.array(Image.open(imgpath))
        red = img[:, :, 0]

        redLin = ImageLinearizationExtractor().extract(red)

        imgLin = ImageLinearizationExtractor().extract(img)

        r = np.array([0]*1024, dtype=np.uint8).reshape(32, 32)
        g = np.array([127]*1024, dtype=np.uint8).reshape(32, 32)
        b = np.array([255]*1024, dtype=np.uint8).reshape(32, 32)

        img2 = np.dstack((r, g, b))

        img2Lin = ImageLinearizationExtractor().extract(img2)

        r2 = np.array([1]*1024, dtype=np.uint8).reshape(32, 32)
        g2 = np.array([128]*1024, dtype=np.uint8).reshape(32, 32)
        b2 = np.array([250]*1024, dtype=np.uint8).reshape(32, 32)

        img3 = np.dstack((r2, g2, b2))

        ls = [img2, img3]

        imgLs = ImageLinearizationExtractor()(ls)
