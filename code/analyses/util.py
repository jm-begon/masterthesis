# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:39:09 2014

@author: Jm Begon
"""
import pylab as pl
import numpy as np

def featureImportanceIndices(size, originalImage):
    if originalImage is not None:
        indices = []
        for i in xrange(size):
            if i == originalImage:
                indices.append(str(i) + " (image)")
            else:
                #indices.append("filter "+str(i))
                indices.append(str(i))
        return indices
    else:
        #return [("filter "+str(x)) for x in xrange(size)]
        return xrange(size)


def plotFeatureImportance(featureImportance, title, originalImage=None, lim=0.06, colorate=None):
    """
    originalImage : the index of the original image. If None, ignore
    """
    indices = featureImportanceIndices(len(featureImportance), originalImage)
    pl.figure()
    pl.title(title)
    if colorate is not None:
        nbType = len(colorate)
        X = [[] for i in range(nbType)]
        Y = [[] for i in range(nbType)]
        for j, f in enumerate(featureImportance):
            X[j % nbType].append(j)
            Y[j % nbType].append(f)
        for i in range(nbType):
            pl.bar(X[i], Y[i], align="center", label=colorate[i][0], color=colorate[i][1])
        pl.legend()
    else:
        pl.bar(range(len(featureImportance)), featureImportance, align="center")
    #pl.xticks(pl.arange(len(indices)), indices, rotation=-90)
    pl.xlim([-1, len(indices)])
    pl.ylabel("Feature importance")
    pl.xlabel("Filter indices")
    pl.ylim(0, lim)
    pl.show()


def getCumulFreq(data):
    indices = np.argsort(data)[::-1]
    data = np.array(data)[indices]
    cumul = np.zeros((len(data)+1))
    for i in xrange(1, len(data)+1):
        cumul[i] = cumul[i-1] + data[i-1]
    return cumul


def readyArray(array, decimal=7):
    print("["),
    for el in array:
        print("%.7f," % el),
    print("]\n"),