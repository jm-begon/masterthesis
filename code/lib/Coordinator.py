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
from matplotlib.cbook import flatten

from Logger import Progressable
from TaskManager import SerialExecutor, ParallelExecutor
from NumpyToPILConvertor import NumpyPILConvertor

__all__ = ["PixitCoordinator", "RandConvCoordinator",
           "CompressRandConvCoordinator"]


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

    def __init__(self, logger=None, verbosity=None):
        Progressable.__init__(self, logger, verbosity)
        self._exec = SerialExecutor(logger, verbosity)

    def parallelize(self, nbJobs=-1, tempFolder=None):
        """
        Parallelize the coordinator

        Parameters
        ----------
        nbJobs : int {>0, -1} (default : -1)
            The parallelization factor. If "-1", the maximum factor is used
        tempFolder : filepath (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :lib:`joblib` library)
        """
        self._exec = ParallelExecutor(nbJobs, self.getLogger(),
                                      self.verbosity, tempFolder)

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

        Implementation
        --------------
        The method :meth:`process` only "schedule" the work. The
        implementation of what is to be done is the responbility of the method
        :meth:`_onProcess`. It is this method that should be overloaded
        """
        ls = self._exec("Extracting features", self._onProcess, imageBuffer)

        # Reduce
        self.logMsg("Concatenating the data...", 35)
        y = np.concatenate([y for _, y in ls])
        X = np.vstack(X for X, _ in ls)

        return X, y

    @abstractmethod
    def _onProcess(self, imageBuffer):
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
    def __init__(self, multiSWExtractor, featureExtractor, logger=None,
                 verbosity=None):
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
        Coordinator.__init__(self, logger, verbosity)
        self._multiSWExtractor = multiSWExtractor
        self._featureExtractor = featureExtractor

    def _onProcess(self, imageBuffer):
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

    def __init__(self, convolutionalExtractor, featureExtractor,
                 logger=None, verbosity=None):
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
        Coordinator.__init__(self, logger, verbosity)
        self._convolExtractor = convolutionalExtractor
        self._featureExtractor = featureExtractor

    def _onProcess(self, imageBuffer):
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

    def getFilters(self):
        """
        Return the filters used to process the image

        Return
        ------
        filters : iterable of numpy arrays
            The filters used to process the image, with the exclusion
            of the identity filter if the raw image was included
        """
        return self._convolExtractor.getFilters()

    def isImageIncluded(self):
        """
        Whether the raw image was included

        Return
        ------
        isIncluded : boolean
            True if the raw image was included
        """
        return self._convolExtractor.isImageIncluded()

    def _groupsInfo(self, nbFeatures):
        """
        Return information about the grouping of features (original image
        included if necessary)

        Parameters
        ----------
         nbFeatures : int > 0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, nbFeaturePerGroup)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        nbFeaturePerGroup : int
            The number of features per group
        """
        nbGroups = len(self.getFilters())
        if self.isImageIncluded:
            nbGroups += 1
        nbFeaturePerGroup = nbFeatures // nbGroups
        return nbFeatures, nbGroups, nbFeaturePerGroup

    def featureGroups(self, nbFeatures):
        """
        Returns an iterable of start indices of the feature groups of X and
        the number of features

        Parameters
        ----------
        nbFeatures : int > 0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, ls)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        ls : iterable of int
            Returns an iterable of start indices of the feature groups of X
            and the number of features
        """
        nbFeatures, nbGroups, nbFeaturePerGroup = self._groupsInfo(nbFeatures)
        return (nbFeatures, nbGroups, xrange(0, nbFeatures+1,
                                             nbFeaturePerGroup))

    def importancePerFeatureGrp(self, classifier):
        """
        Computes the importance of each filter.

        Parameters
        ----------
        classifier : sklearn.ensemble classifier with
        :attr:`feature_importances_`
            The classifier (just) used to fit the model
        X : 2D numpy array
            The feature array. It must have been learnt by this
            :class:`ConvolutionalExtractor` with the given classifier
        Return
        ------
        pair = (importance, indices)
        importance : iterable of real
            the importance of each group of feature
        indices : iterable of int
            the sorted indices of importance in decreasing order
        """

        importance = classifier.feature_importances_
        nbFeatures, nbGroups, starts = self.featureGroups(len(importance))
        impPerGroup = []
        for i in xrange(nbGroups):
            impPerGroup.append(sum(importance[starts[i]:starts[i+1]]))

        return impPerGroup, np.argsort(impPerGroup)[::-1]


class CompressRandConvCoordinator(RandConvCoordinator):
    """
    ===========================
    CompressRandConvCoordinator
    ===========================
    This :class:`RandConvCoordinator` class adds the feature of compressing
    the extracted features.
    The compression operates on each filter ouput separately and depends
    on the given :class:`Compressor`.
    It is possible not to include the first image in the compression
    """
    def __init__(self, convolutionalExtractor, featureExtractor, compressor,
                 compressOriginalImage=True, logger=None, verbosity=None):
        """
        Construct a :class:`CompressRandConvCoordinator`

        Parameters
        ----------
        convolutionalExtractor : :class:`ConvolutionalExtractor`
            The component responsible for the extraction, filtering and
            aggregation of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each filtered and aggregated subwindow
        compressor : :class:`Compressor`
            The component responsible for compressing the feature vector
            filterwise.

        Note
        ----
        The :class:`FeatureExtractor` instance must be adequate wrt the image
        type
        """
        RandConvCoordinator.__init__(self, convolutionalExtractor,
                                     featureExtractor, logger, verbosity)
        self._compressor = compressor
        self._compressImage = compressOriginalImage

    def process(self, imageBuffer):
        imgIncluded = self._convolExtractor.isImageIncluded()
        X, y = RandConvCoordinator.process(self, imageBuffer)
        data = self._slice(X)
        ls = self._exec(self._compressor.compress, data, y)
        X2 = np.hstack(ls)
        if imgIncluded and not self._compressImage:
            _, _, endImage = self._groupsInfo(X)
            imgX = X.tranpose()[0:endImage]
            X2 = np.hstack((imgX.transpose(), X2))
        return X2, y

    def _slice(self, X):
        """
        Slice the feature array appropriately for the compression

        Parameters
        ----------
        X : 2D numpy array
            The feature array
        Return
        ------
        A iterable subarray of the feature array as group of features generated
        by the same filter
        """
        XTranspose = X.transpose()  # Only a view
        slices = []
        nbFeatures, nbGroups, nbFeaturePerGroup = self._groupsInfo(X)
        imgIncluded = self._convolExtractor.isImageIncluded()
        for i in xrange(nbGroups):
            Xtmp = XTranspose[i*nbFeaturePerGroup:(i+1)*nbFeaturePerGroup]
            slices.append(Xtmp.transpose())
        if imgIncluded and not self._compressImage:
            slices = slices[1:]
            self._imgSize = nbFeaturePerGroup
        return slices

    def featureGroups(self, nbFeatures):
        """
        Returns an iterable of start indices of the feature groups of X and
        the number of features

        Parameters
        ----------
        nbFeatures : int >0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, ls)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        ls : iterable of int
            Returns an iterable of start indices of the feature groups of X
            and the number of features
        """
        if (not self._compressImage) and self._convolExtractor.isImageIncluded():
            nbGroups = self.getFilters()  # Discouting the image
            nbFeaturePerGroup = (nbFeatures - self._imgSize) // nbGroups
            ls = [0]
            starts = xrange(self._imgSize, nbFeatures+1, nbFeaturePerGroup)
            return (nbFeatures, nbGroups+1, nbFeaturePerGroup,
                    flatten([ls, starts]))
        else:
            return RandConvCoordinator.featureGroups(self, X)

#TODO XXX overload _grpInfo

if __name__ == "__main__":
    def _slice(X, nbFilter, imgIncluded, imgCompressed):
        XTranspose = X.transpose()
        slices = []
        nbGroups = nbFilter
        nbFeatures = X.shape[1]
        nbFeatPerGrp = nbFeatures / nbGroups
        for i in xrange(nbGroups):
            Xtmp = XTranspose[i*nbFeatPerGrp:(i+1)*nbFeatPerGrp]
            slices.append(Xtmp.transpose())
        if imgIncluded and not imgCompressed:
            slices = slices[1:]
        return slices

    X = np.arange(30).reshape(5, 6)
    print X
    ls = _slice(X, 3, False, False)
    for x in ls:
        print x
