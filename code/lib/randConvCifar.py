# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the random and convolution classifcation
"""
from time import time

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
    tempFolder = "temp/"

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
    maxLearningSize = 50000
    learningUse = 100
    learningSetDir = "learn/"
    learningIndexFile = "0index"
    maxTestingSize = 10000
    testingUse = 100
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
        verbosity=verbosity,
        tempFolder=tempFolder)

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
    learningSet[0:learningUse]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet[0:testingUse]

    #=====COMPUTATION=====#
    #--Learning--#
    fitStart = time()
    classifier.fit(learningSet)
    fitEnd = time()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)

    print ">>>>>First 50<<<<<<"
    for i in xrange(min((50, len(y_pred)))):
        print y_pred[i], y_truth[i]

    print "========================================="
    print "-----------Filtering--------------"
    print "nb_filters", nb_filters
    print "filter_min_val", filter_min_val
    print "filter_max_val", filter_max_val
    print "filterMinSize", filterMinSize
    print "filterMaxSize", filterMaxSize
    print "filterNormalisation", filterNormalisation
    print "----------Pooling--------------"
    print "aggregatorNeighborhoodWidth", aggregatorNeighborhoodWidth
    print "aggregatorNeighbordhoodHeight", aggregatorNeighbordhoodHeight
    print "--------SW extractor----------"
    print "#Subwindows", nbSubwindows
    print "subwindowMinSizeRatio", subwindowMinSizeRatio
    print "subwindowMaxSizeRatio", subwindowMaxSizeRatio
    print "subwindowTargetWidth", subwindowTargetWidth
    print "subwindowTargetHeight", subwindowTargetHeight
    print "fixedSize", fixedSize
    print "------------Misc-----------------"
    print "includeOriginalImage", includeOriginalImage
    print "nbJobs", nbJobs
    print "--------ExtraTrees----------"
    print "nbTrees", nbTrees
    print "maxFeatures", maxFeatures
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
    print "bootstrap", bootstrap
    print "nbJobsEstimator", nbJobsEstimator
    print "randomState", randomState
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "Fit time", (fitEnd-fitStart), "seconds"
    print "Fit time", (predEnd-predStart), "seconds"
    print "Accuracy", accuracy
