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

from CoordinatorFactory import Const, coordinatorRandConvFactory
from Classifier import Classifier
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
poolings = [(2, 2, Const.POOLING_AGGREG_AVG)]

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

#-----Extratree param
nbTrees = 30
maxFeatures = "auto"
maxDepth = None
minSamplesSplit = 2
minSamplesLeaf = 1
bootstrap = False
nbJobsEstimator = -1
verbose = 8

#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learningUse = 100
learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 100
testingSetDir = "test/"
testingIndexFile = "0index"


def run(nb_filters=nb_filters,
        filter_min_val=filter_min_val,
        filter_max_val=filter_max_val,
        filterMinSize=filterMinSize,
        filterMaxSize=filterMaxSize,
        filterNormalisation=filterNormalisation,
        poolings=poolings,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        fixedSize=fixedSize,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        random=random,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        nbTrees=nbTrees,
        maxFeatures=maxFeatures,
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minSamplesLeaf=minSamplesLeaf,
        bootstrap=bootstrap,
        nbJobsEstimator=nbJobsEstimator,
        verbose=verbose,
        learningUse=learningUse,
        testingUse=testingUse):

    randomState = None
    if random:
        randomState = 100

    lsSize = learningUse
    if learningUse > maxLearningSize:
        lsSize = maxLearningSize

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

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
        poolings=poolings,
        filterNormalisation=filterNormalisation,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        random=random)

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
    learningSet = learningSet[0:lsSize]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

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
    confMat = classifier.confusionMatrix(y_pred, y_truth)

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
    print "poolings", poolings
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
    print "--------ExtraTrees----------"
    print "nbTrees", nbTrees
    print "maxFeatures", maxFeatures
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
    print "bootstrap", bootstrap
    print "nbJobsEstimator", nbJobsEstimator
    print "verbose", verbose
    print "randomState", randomState
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "Fit time", (fitEnd-fitStart), "seconds"
    print "Classifcation time", (predEnd-predStart), "seconds"
    print "Accuracy", accuracy

    return accuracy, confMat, importance, order

if __name__ == "__main__":
    acc, confMat, importance, order = run(nb_filters=10)

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
