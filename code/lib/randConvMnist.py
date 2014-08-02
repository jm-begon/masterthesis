# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the random and convolution classifcation
"""
import sys
import numpy as np
from time import time

from sklearn.ensemble import ExtraTreesClassifier

import mnist

from CoordinatorFactory import Const, coordinatorRandConvFactory
from Classifier import Classifier
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader
from Logger import formatDuration
from FilterGenerator import Finite3SameFilter

#======PROB MATRIX=========#
saveFile = "rc_"+str(time())
shouldSave = False

#======MISC.======#
shouldPrintFilters = False
nbFilterOptimization = 0
#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 0
#filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":10, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})
#
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})

#Aggregation
poolings = [
        (2, 2, Const.POOLING_AGGREG_AVG),
#        (3, 3, Const.POOLING_MW_AVG),
#        (3, 3, Const.POOLING_MW_MIN),
#        (3, 3, Const.POOLING_MW_MAX)
        ]

#Extraction
extractor = (Const.FEATEXT_ALL, {})
#extractor =  (Const.FEATEXT_SPASUB, {"nbCol":2})
#extractor =  (Const.FEATEXT_SPASUB, {"nbCol":1})

#Subwindow
nbSubwindows = 10
subwindowMinSizeRatio = 0.1
subwindowMaxSizeRatio = 0.75
subwindowTargetWidth = 16
subwindowTargetHeight = 16
fixedSize = False
subwindowInterpolation = SubWindowExtractor.INTERPOLATION_NEAREST

#Misc.
includeOriginalImage = True
random = False
nbJobs = -1
verbosity = 40
#tempFolder = "/dev/shm"
#tempFolder = "/home/jmbegon/jmbegon/code/work/tmp"
tempFolder = None

#-----Extratree param
nbTrees = 30
maxFeatures = "auto"
maxDepth = None
minSamplesSplit = 2
minSamplesLeaf = 1
bootstrap = False
randomClassif = True
nbJobsEstimator = -1
verbose = 8


#
shufflingRate = -1

def run(nb_filters=nb_filters,
        filterPolicy=filterPolicy,
        poolings=poolings,
        extractor=extractor,
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
        randomClassif=randomClassif,
        nbJobsEstimator=nbJobsEstimator,
        verbose=verbose,
        learningUse=learningUse,
        testingUse=testingUse,
        saveFile=saveFile,
        shouldSave=shouldSave,
        nbFilterOptimization=nbFilterOptimization,
        shufflingRate=shufflingRate):

    randomState = None
    if not randomClassif:
        randomState = 100

    lsSize = learningUse
    if learningUse > maxLearningSize:
        lsSize = maxLearningSize

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

    #======INSTANTIATING========#
    #--RandConv--
    randConvCoord = coordinatorRandConvFactory(
        nbFilters=nb_filters,
        filterPolicy=filterPolicy,
        poolings=poolings,
        extractor=extractor,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        random=random)

    nb_filters = len(randConvCoord.getFilters())

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
    train_set_imgs, train_set_labels, test_set_imgs, test_set_labels = mnist.loadMnist()

    learningSet = ShufflerImageBuffer(ImageBuffer(zip(train_set_imgs, train_set_labels), ImageBuffer.NUMPY_FORMAT), 28, 28, shufflingRate)

    testingSet = ShufflerImageBuffer(ImageBuffer(zip(test_set_imgs, test_set_labels), ImageBuffer.NUMPY_FORMAT), 28, 28, shufflingRate)


    #=====COMPUTATION=====#
    #------Filter optimization-----#
    if nbFilterOptimization > 0:
        randConvOptimizer = coordinatorRandConvFactory(
            nbFilters=nbFilterOptimization,
            filterPolicy=filterPolicy,
            poolings=poolings,
            extractor=(Const.FEATEXT_SPASUB, {"nbCol":1}),
            nbSubwindows=5,
            subwindowMinSizeRatio=subwindowMinSizeRatio,
            subwindowMaxSizeRatio=subwindowMaxSizeRatio,
            subwindowTargetWidth=subwindowTargetWidth,
            subwindowTargetHeight=subwindowTargetHeight,
            subwindowInterpolation=subwindowInterpolation,
            includeOriginalImage=includeOriginalImage,
            nbJobs=nbJobs,
            verbosity=verbosity,
            tempFolder=tempFolder,
            random=random)

        totallyTrees = ExtraTreesClassifier(30,
                                            max_features=1,
                                            max_depth=maxDepth,
                                            min_samples_split=minSamplesSplit,
                                            min_samples_leaf=minSamplesLeaf,
                                            bootstrap=bootstrap,
                                            n_jobs=nbJobsEstimator,
                                            random_state=randomState,
                                            verbose=verbose)

        optiClassif = Classifier(randConvCoord, baseClassif)
        print "Starting optimization"
        optiStart = time()
        optiClassif.fit(learningSet)
        optiEnd = time()
        print "optimization done", formatDuration(optiEnd-optiStart)
        _, order = randConvOptimizer.importancePerFeatureGrp(totallyTrees)

        filtersTmp = randConvCoord._convolExtractor._finiteFilter._filters
        filters = [x for x, _, _ in filtersTmp]
        if not includeOriginalImage:
            bestIndices = order[:nb_filters]
        else:
            count = 0
            bestIndices = []
            for index in order:
                if count == nb_filters-1:
                    break
                if index != 0:
                    bestIndices.append(index-1)
                    count += 1
        bestFlters = []
        for i in bestIndices:
            bestFlters.append(filters[i])

        best3Filters = Finite3SameFilter(bestFlters)
        randConvCoord._convolExtractor._finiteFilter = best3Filters


    #--Learning--#
    print "Starting learning"
    fitStart = time()
    classifier.fit(learningSet)
    fitEnd = time()
    print "Learning done", formatDuration(fitEnd-fitStart)
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_prob, y_pred = classifier.predict_predict_proba(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusionMatrix(y_pred, y_truth)

    #====ANALYSIS=====#
    importance, order = randConvCoord.importancePerFeatureGrp(baseClassif)

    print "==================RandConv================"
    print "-----------Filtering--------------"
    print "nb_filters", nb_filters
    print "filterPolicy", filterPolicy
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
    if shouldSave:
        print "saveFile", saveFile
    print "Fit time", formatDuration(fitEnd-fitStart)
    print "Classifcation time", formatDuration(predEnd-predStart)
    print "Accuracy", accuracy

    if shouldSave:
        np.save(saveFile, y_prob)

    filtersTmp = randConvCoord._convolExtractor._finiteFilter._filters
    filters = [x for x, _, _ in filtersTmp]

    return accuracy, confMat, importance, order, filters


if __name__ == "__main__":

    acc, confMat, importance, order, filters  = run()

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
    if shouldPrintFilters:
        print "10 bests filters"
        for i in xrange(10):
            if order[i] == 0 and includeOriginalImage:
                print "id filter"
            else:
                print filters[order[i]]
                print ""
        print "========="
        print "10 worst filters"
        for i in xrange(10):
            if order[-i] == 0 and includeOriginalImage:
                print "id filter"
            else:
                print filters[order[-i]]
                print ""


