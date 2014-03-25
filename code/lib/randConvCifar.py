# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the random and convolution classifcation
"""
import sys
import os
from time import time

from sklearn.ensemble import ExtraTreesClassifier

from CoordinatorFactory import coordinatorRandConvFactory
from Classifier import Classifier
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader


def run():

    #======HYPER PARAMETERS======#
    #----RandConv param
    #Filtering
    nb_filters = 100
    filter_min_val = -5
    filter_max_val = 5
    filterMinSize = 3
    filterMaxSize = 6
    filterNormalisation = FilterGenerator.NORMALISATION_NONE

    #Aggregation
    aggregatorNeighborhoodWidth = 2
    aggregatorNeighbordhoodHeight = 2

    #Subwindow
    nbSubwindows = 10
    subwindowMinSizeRatio = 0.75
    subwindowMaxSizeRatio = 1.
    subwindowTargetWidth = 16
    subwindowTargetHeight = 16
    fixedSize = False
    subwindowInterpolation = SubWindowExtractor.INTERPOLATION_BILINEAR

    #Misc.
    includeOriginalImage = True
    nbJobs = -1
    verbosity = 7
    tempFolder = "temp/"

    #-----Extratree param
    nbTrees = 30
    maxFeatures = "auto"
    maxDepth = None
    minSamplesSplit = 2
    minSamplesLeaf = 1
    bootstrap = False
    nbJobsEstimator = -1
    randomState = None
    verbose = 8

    #=====DATA=====#
#    maxLearningSize = 50000
#    maxTestingSize = 10000

    learningUse = 500
    learningSetDir = "learn/"
    learningIndexFile = "0index"

    testingUse = 500
    testingSetDir = "test/"
    testingIndexFile = "0index"

    #======INSTANTIATING========#
    os.environ["JOBLIB_TEMP_FOLDER"] = "/home/jmbegon/jmbegon/code/work/tmp/"
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
    learningSet = learningSet[0:learningUse]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:testingUse]

    #=====COMPUTATION=====#
    #--Learning--#
    print "Starting learning"
    fitStart = time()
    classifier.fit(learningSet)
    fitEnd = time()
    print "Learning done", (fitEnd-fitStart), "seconds"
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)

    #====ANALYSIS=====#
    importance, order = randConvCoord.importancePerFeatureGrp(baseClassif)

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
    print "Classifcation time", (predEnd-predStart), "seconds"
    print "Accuracy", accuracy

    return accuracy, importance, order

if __name__ == "__main__":
    acc, importance, order = run()

    print importance
    print order
