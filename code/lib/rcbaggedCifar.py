# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 19 2014
"""
A script to run the random and convolution classifcation
"""
import sys
from time import time

from sklearn.svm import LinearSVC

from CoordinatorFactory import Const, coordinatorRandConvFactory
from Classifier import UnsupervisedVisualBagClassifier as uClassifier
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader


#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 100
filterPolicy = (Const.FGEN_CUSTOM, {"normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_MEANVAR})
#filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_MEANVAR})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":minPerturbation, "maxPerturbation":maxPerturbation})
#
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_MEANVAR})
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_MEANVAR})
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_MEANVAR})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":minPerturbation, "maxPerturbation":maxPerturbation})

#Aggregation
poolings = [(2, 2, Const.POOLING_MW_AVG)]

#Subwindow
nbSubwindows = 20
subwindowMinSizeRatio = 0.75
subwindowMaxSizeRatio = 1.
subwindowTargetWidth = 16
subwindowTargetHeight = 16
fixedSize = False
subwindowInterpolation = SubWindowExtractor.INTERPOLATION_NEAREST

#Misc.
includeOriginalImage = True
random = False
nbJobs = -1
verbosity = 8
tempFolder = "/dev/shm"

#-----BagOfWords params + some SVC params
nbTrees = 30
maxDepth = None  # 5
minSamplesSplit = 2
minSamplesLeaf = 1
nbJobsEstimator = -1
verbose = 8
#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learningUse = 50
learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 10
testingSetDir = "test/"
testingIndexFile = "0index"


def run(nb_filters=nb_filters,
        filterPolicy=filterPolicy,
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
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minSamplesLeaf=minSamplesLeaf,
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
    #--randconv--
    randConvCoord = coordinatorRandConvFactory(
        nbFilters=nb_filters,
        filterPolicy=filterPolicy,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        poolings=poolings,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        random=random)

    nb_filters = len(randConvCoord.getFilters())

    #--Extra-tree--
    baseClassif = LinearSVC(verbose=verbose, random_state=randomState)

    #--Classifier
    classifier = uClassifier(coordinator=randConvCoord,
                             base_classifier=baseClassif,
                             n_estimators=nbTrees,
                             max_depth=maxDepth,
                             min_samples_split=minSamplesSplit,
                             min_samples_leaf=minSamplesLeaf,
                             n_jobs=nbJobsEstimator,
                             random_state=randomState,
                             verbose=verbose)

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
    importance, order = randConvCoord.importancePerFeatureGrp(classifier._visualBagger)

    print "==================Bag of Visual Words======================="
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
    print "--------Bag of words params + SVC----------"
    print "nbTrees", nbTrees
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
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
    acc, confMat, importance, order = run()

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
