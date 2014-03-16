# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
:class:`Coordinator` are responsible for applying a feature extraction
mechanism to all the data contained in a imageBuffer and keeping the
consistency if it creates several feature vectors for one image
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from Logger import Progressable
from NumpyToPILConvertor import NumpyPILConvertor

__all__ = ["PixitCoordinator", "RandConvCoordinator"]


class Coordinator(Progressable):
    """
    ===========
    Coordinator
    ===========

    :class:`Coordinator` are responsible for applying a feature extraction
    mechanism to all the data contained in a imageBuffer and keeping the
    consistency if it creates several feature vectors for one image.

    The extraction mechanism is class dependent. It is the class
    responsability to document its policy.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        Progressable.__init__(self)

    @abstractmethod
    def process(self, imageBuffer):
        """
        Extracts the feature vectors for the images contained in the
        :class:`ImageBuffer`

        Abstract method to overload.

        Parameters
        ----------
        imageBuffer : :class:`ImageBuffer`
            The data to process

        Return
        ------
        X : a numpy 2D array
            the N x M feature matrix. Each of the N rows correspond to an
            object and each of the M columns correspond to a variable
        y : an iterable of int
            the N labels corresponding to the N objects of X

        Note
        ----
        The method might provide several feature vectors per original image.
        It ensures the consistency with the labels and is explicit about
        the mapping.
        """
        pass

    def __call__(self, imageBuffer):
        """Delegate to :meth:`process`"""
        return self.process(imageBuffer)

    def getLogger(self):
        """
        Return
        ------
        logger : :class:`Logger`
            The internal logger (might be None)
        """
        return self._logger


class PixitCoordinator(Coordinator):
    """
    ================
    PixitCoordinator
    ================

    This coordinator uses a :class:`MultiSWExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows
    from the image while the second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`MultiSWExtractor` instance but are grouped
    contiguously.

    Note
    ----
    The :class:`FeatureExtractor` instance must be adequate wrt the image
    type
    """
    def __init__(self, multiSWExtractor, featureExtractor):
        """
        Construct a :class:`PixitCoordinator`

        Parameters
        ----------
        multiSWExtractor : :class:`MultiSWExtractor`
            The component responsible for the extraction of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each subwindow
        """
        Coordinator.__init__(self)
        self._multiSWExtractor = multiSWExtractor
        self._featureExtractor = featureExtractor

    def process(self, imageBuffer):
        """Overload"""
        ls = []
        y = []
        convertor = NumpyPILConvertor()
        
        #Logging
        counter = 0
        self.setTask(len(imageBuffer),
                     "PixitCoordinator loop for each image")
        for image, label in imageBuffer:
            image = convertor.numpyToPIL(image)
            imgLs = self._multiSWExtractor.extract(image)
            for img in imgLs:
                ls.append(
                    self._featureExtractor.extract(convertor.pILToNumpy(img)))
            y = y + [label] * len(imgLs)
            #Logging progress
            self.updateTaskProgress(counter)
            counter += 1
        X = np.vstack((ls))

        return X, y


class RandConvCoordinator(Coordinator):
    """
    ===================
    RandConvCoordinator
    ===================

    This coordinator uses a :class:`ConvolutionalExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows from
    the image applies filter to each subwindow and aggregate them while the
    second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`ConvolutionalExtractor` instance but are grouped
    contiguously.
    """

    def __init__(self, convolutionalExtractor, featureExtractor):
        """
        Construct a :class:`RandConvCoordinator`

        Parameters
        ----------
        convolutionalExtractor : :class:`ConvolutionalExtractor`
            The component responsible for the extraction, filtering and
            aggregation of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each filtered and aggregated subwindow

        Note
        ----
        The :class:`FeatureExtractor` instance must be adequate wrt the image
        type
        """
        Coordinator.__init__(self)
        self._convolExtractor = convolutionalExtractor
        self._featureExtractor = featureExtractor

    def process(self, imageBuffer):
        """Overload"""
        ls = []
        y = []

        #Logging
        counter = 0
        self.setTask(len(imageBuffer),
                     "RandConvCoordinator loop for each image")

        for image, label in imageBuffer:
            #Get the subwindows x filters
            allSubWindows = self._convolExtractor.extract(image)

            #Accessing each subwindow sets separately
            for filteredList in allSubWindows:
                filtersBySW = []
                #Accessing each filter separately for a given subwindow
                for filtered in filteredList:
                    #Extracting the features for each filter
                    filtersBySW.append(self._featureExtractor.extract(filtered))
                #Combining all the filter_feature for a given subwindow
                ls.append(np.hstack(filtersBySW))

            #Extending the labels for each subwindow
            y = y + [label]*len(allSubWindows)

            #Logging progress
            self.updateTaskProgress(counter)
            counter += 1
        #Combining the information for all the images
        X = np.vstack((ls))

        return X, y
