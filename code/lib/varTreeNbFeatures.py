# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 14:40:31 2014

@author: Jm Begon

Repeat randConv with different parameters
"""
import cPickle
import randConvCifar as rc


if __name__ == "__main__":
    fileName = "varTreeFeatures"
#    nbTree = [10, 60, 120, 500]
#    nbFeatures = [100, 500, 1000, 5000]
    nbTree = [10]
    nbFeatures = [10]
    includeOriginalImage = True

    tupTree = []
    tupFeat = []
    for i in range(min([len(nbTree), len(nbFeatures)])):
        #Var nb tree
        tup = rc.run(includeOriginalImage=includeOriginalImage,
                     nbTrees=nbTree[i],
                     random=False)
        fn = fileName+"_tree"+str(nbTree[i])
        with open(fn, "wb") as f:
            cPickle.dump(tup, f, protocol=2)
        tupTree.append(tup)
        #Var nb features
        tup = rc.run(includeOriginalImage=includeOriginalImage,
                     maxFeatures=nbFeatures[i],
                     random=False)
        fn = fileName+"_feature"+str(nbFeatures[i])
        with open(fn, "wb") as f:
            cPickle.dump(tup, f, protocol=2)
        tupFeat.append(tup)
        #Signal
        print ">>>> ", i, " done"
        print tupTree
        print tupFeat
