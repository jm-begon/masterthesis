# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 31 2014
"""
A script to run the random and convolution classifcation
"""
import sys
import os
from time import time
import cPickle as pickle


from CoordinatorFactory import coordinatorRandConvFactory
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader

#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 100
filter_min_val = -1
filter_max_val = 1
filterMinSize = 2
filterMaxSize = 32
filterNormalisation = FilterGenerator.NORMALISATION_MEANVAR

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
random = False
nbJobs = -1
verbosity = 8
tempFolder = "tmp/"

#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learningUse = 500
learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 500
testingSetDir = "test/"
testingIndexFile = "0index"


def run(lsName, tsName, **kwargs):

    lsSize = learningUse
    if learningUse > maxLearningSize:
        lsSize = maxLearningSize

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

    #======INSTANTIATING========#
    os.environ["JOBLIB_TEMP_FOLDER"] = "/home/jmbegon/jmbegon/code/work/tmp/"

    #--Coordinator--
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
        tempFolder=tempFolder,
        random=random)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:lsSize]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

    #=====COMPUTATION=====#
    #--Learning--#
    print "Starting learning"
    fitStart = time()
    X, y = randConvCoord.process(learningSet, True)
    with open(lsName, "wb") as f:
            pickle.dump((lsSize, X, y), f, protocol=2)
    fitEnd = time()
    print "Learning done", (fitEnd-fitStart), "seconds"
    sys.stdout.flush()

    #--Testing--#
    predStart = time()
    X, y = randConvCoord.process(testingSet, False)
    with open(tsName, "wb") as f:
            pickle.dump((tsSize, X, y), f, protocol=2)
    predEnd = time()

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
    print "random", random
    print "tempFolder", tempFolder
    print "verbosity", verbosity
    print "nbJobs", nbJobs
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "Fit time", (fitEnd-fitStart), "seconds"
    print "Classifcation time", (predEnd-predStart), "seconds"


if __name__ == "__main__":
    acc, confMat, importance, order = run()

    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
