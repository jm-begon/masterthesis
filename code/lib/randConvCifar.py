# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the random and convolution classifcation
"""
from sklearn.ensemble import ExtraTreesClassifier

from CoordinatorFactory import coordinatorRandConvFactory
from Classifier import Classifier
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader

if __name__ == "__main__":

    #======HYPER PARAMETERS======#
    #----RandConv param
    #Filtering
    nb_filters = 10
    filter_min_val = -1
    filter_max_val = 1
    filterMinSize = 3
    filterMaxSize = 32
    filterNormalisation = FilterGenerator.NORMALISATION_NONE  # TODO

    #Aggregation
    aggregatorNeighborhoodWidth = 2
    aggregatorNeighbordhoodHeight = 2

    #Subwindow
    nbSubwindows = 10
    subwindowMinSizeRatio = 0.5
    subwindowMaxSizeRatio = 1.
    subwindowTargetWidth = 16
    subwindowTargetHeight = 16
    fixedSize = False
    subwindowInterpolation = SubWindowExtractor.INTERPOLATION_BILINEAR

    #Misc.
    includeOriginalImage = True
    nbJobs = 1
    verbosity = 10

    #-----Extratree param
    nbTrees = 10
    maxFeatures = "auto"
    maxDepth = None
    minSamplesSplit = 2
    minSamplesLeaf = 1
    bootstrap = False
    nbJobsEstimator = -1
    randomState = None
    verbose = 5

    #=====DATA=====#
    learningSetDir = "learn/"
    learningIndexFile = "0index"
    testingSetDir = "test/"
    testingIndexFile = "0index"

    #======INSTANTIATING========#
    #--Pixit--
    randConvCoord = coordinatorRandConvFactory(
        nbFilters=nb_filters,
        filterMinVal=filter_min_val,
        filterMaxVal=filter_max_val,
        filterMinSize=filterMinSize,
        filterMaxSize=filterMaxSize,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        aggregatorNeighborhoodWidth=aggregatorNeighborhoodWidth,
        aggregatorNeighbordhoodHeight=aggregatorNeighbordhoodHeight,
        filterNormalisation=filterNormalisation,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        nbJobs=nbJobs,
        verbosity=verbosity)

    #--Extra-tree--
    baseClassif = ExtraTreesClassifier(nbTrees,
                                       max_features=maxFeatures,
                                       max_depth=maxDepth,
                                       min_samples_split=minSamplesSplit,
                                       min_samples_leaf=minSamplesLeaf,
                                       bootstrap=bootstrap,
                                       n_jobs=nbJobsEstimator,
                                       random_state=randomState,
                                       verbose=verbose)

    #--Classifier
    classifier = Classifier(randConvCoord, baseClassif)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())

    #=====COMPUTATION=====#
    #--Learning--#
    classifier.fit(learningSet)

    #--Testing--#
    y_truth = testingSet.getLabels()
    y_pred = classifier.predict(testingSet)
    accuracy = classifier.accuracy(y_pred, y_truth)

    print y_truth
    print y_pred
    print "========================================="
    print "Accuracy:\t", accuracy

    print classifier.predict_proba(testingSet)
