# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:04:05 2014

@author: Jm
"""

import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from CoordinatorFactory import Const
from Classifier import Classifier
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader
from Coordinator import Coordinator


class MemroyTestCoordinator(Coordinator):
    def __init__(self, nbFeatures, nbObjects):
        self._nbFeatures = nbFeatures
        self._nbObj = nbObjects

    def _onProcess(self, imageBuffer=None, learningPhase=None):
        X = np.zeros((self._Obj, self._nbFeatures))
        y = [0]*self._nbObj
        return X, y


#======HYPER PARAMETERS======#
nb_filters = 100
poolings = [(2, 2, Const.POOLING_AGGREG_AVG)]

nbSubwindows = 10
subwindowTargetWidth = 16
subwindowTargetHeight = 16

nbJobs = 30
verbosity = 50
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

learningUse = 50000
learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 10000
testingSetDir = "test/"
testingIndexFile = "0index"


def run(nb_filters=nb_filters,
        poolings=poolings,
        nbSubwindows=nbSubwindows,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
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

    lsSize = learningUse
    if learningUse > maxLearningSize:
        lsSize = maxLearningSize

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

    totalNbFeatures = nb_filters*len(poolings)*subwindowTargetWidth*subwindowTargetHeight*3
    totalNbObj = lsSize*nbSubwindows

    nbFeatures = totalNbFeatures/nbJobs

    floatSize = np.zeros().itemsize
    singleArraySize = nbFeatures*totalNbObj*floatSize
    totalArraySize = totalNbFeatures*totalNbObj*floatSize

    #======INSTANTIATING========#
    os.environ["JOBLIB_TEMP_FOLDER"] = "/home/jmbegon/jmbegon/code/work/tmp/"
    #--Pixit--
    memCoord = MemroyTestCoordinator(nbFeatures, totalNbObj)
    if nbJobs != 1:
        memCoord.parallelize(nbJobs, tempFolder)

    #--Extra-tree--
    baseClassif = ExtraTreesClassifier(nbTrees,
                                       max_features=maxFeatures,
                                       max_depth=maxDepth,
                                       min_samples_split=minSamplesSplit,
                                       min_samples_leaf=minSamplesLeaf,
                                       bootstrap=bootstrap,
                                       n_jobs=nbJobsEstimator,
                                       verbose=verbose)

    #--Classifier
    classifier = Classifier(memCoord, baseClassif)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:lsSize]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

    #=====COMPUTATION=====#
    #--Learning--#
    classifier.fit(learningSet)

    print "========================================="
    print "-----------Filtering--------------"
    print "nb_filters", nb_filters
    print "----------Pooling--------------"
    print "poolings", poolings
    print "--------SW extractor----------"
    print "#Subwindows", nbSubwindows
    print "subwindowTargetWidth", subwindowTargetWidth
    print "subwindowTargetHeight", subwindowTargetHeight
    print "------------Misc-----------------"
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
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "totalNbFeatures", totalNbFeatures
    print "totalNbObj", totalNbObj
    print "singleArraySize", singleArraySize
    print "totalArraySize", totalArraySize


if __name__ == "__main__":
    run()
