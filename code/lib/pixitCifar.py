# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the pixit classification.
"""

from CoordinatorFactory import coordinatorPixitFactory
from sklearn.ensemble import ExtraTreesClassifier
from Classifier import Classifier
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader


if __name__ == "__main__":

    #======HYPER PARAMETERS======#
    #PixitCoordinator param
    nbSubwindows = 10
    subwindowMinSizeRatio = 0.5
    subwindowMaxSizeRatio = 1.
    subwindowTargetWidth = 16
    subwindowTargetHeight = 16
    fixedSize = False
    nbJobs = -1
    verbosity = 10

    #Extratree param
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
    pixitCoord = coordinatorPixitFactory(nbSubwindows,
                                         subwindowMinSizeRatio,
                                         subwindowMaxSizeRatio,
                                         subwindowTargetWidth,
                                         subwindowTargetHeight,
                                         fixedSize=fixedSize,
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
    classifier = Classifier(pixitCoord, baseClassif)

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

    print "========================================="
    print "Accuracy:\t", accuracy
    
    print classifier.predict_proba(testingSet)
