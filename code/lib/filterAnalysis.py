# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:20:50 2014

@author: Jm Begon
"""
import pylab as pl
from PIL import Image
from scipy import signal as sg
import scipy as sc
import numpy as np


from NumberGenerator import OddUniformGenerator, NumberGenerator
from FilterGenerator import FilterGenerator, Finite3SameFilter


def histoFilters(filters):
    histo = {}
    for filt in filters:
        size = len(filt)
        if size in histo:
            histo[size] += 1
        else:
            histo[size] = 1
    return histo


def plotFilter(filt):
    pl.figure()
    pl.imshow(filt, interpolation="none", vmin=-1, vmax=1, cmap="hot")
    pl.colorbar(extend="both")
    pl.show()


def getLena():
    lena = np.array(Image.open("lena.jpg"))
    height, width, _ = lena.shape
    lena2 = np.zeros((height, width), dtype=np.uint8)
    for h in xrange(height):
        for w in xrange(width):
            lena2[h][w] = lena[h][w][0]
    return lena2


def applyFilter(lena, filt):
    return sg.convolve2d(lena, filt, "same", "fill", 0)


def psd(img):
    tmp = np.abs(sc.fftpack.fftshift(sc.fftpack.fft2(img)))**2
    return tmp.astype(np.uint8)


def plotImage(img):
    pl.figure()
    pl.imshow(img, cmap="gray")
    pl.show()


def delta(imp1, imp2):
    l1 = [0]*len(imp1)
    l2 = [0]*len(imp2)
    for i in xrange(len(imp1)):
        l1[imp1[i]] = i
        l2[imp2[i]] = i
    return [(x-y) for x, y in zip(l1, l2)]


def recap(img, filt):
    filtered = applyFilter(img, filt)
    U = sc.fftpack.fftshift(sc.fftpack.fft2(img))
    Y = sc.fftpack.fftshift(sc.fftpack.fft2(filtered))
    H = Y/U
    Upsd = np.abs(U)**2
    Ypsd = np.abs(Y)**2
    Hpsd = np.abs(H)**2
    return Upsd.astype(np.uint8), Hpsd.astype(np.uint8), Ypsd.astype(np.uint8)


if __name__ == "__main__":

    filtValGenSeed = 1
    filtSizeGenSeed = 2
    filterMinVal = -1
    filterMaxVal = 1
    filterMinSize = 2
    filterMaxSize = 32
    nbFilters = 100
    nbMainFilters = 10
    printFilter = 0
    filterNormalisation = FilterGenerator.NORMALISATION_MEANVAR

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    filterValGenerator = NumberGenerator(filterMinVal, filterMaxVal,
                                         seed=filtValGenSeed)
    filterSizeGenerator = OddUniformGenerator(filterMinSize, filterMaxSize,
                                              seed=filtSizeGenSeed)
    baseFilterGenerator = FilterGenerator(filterValGenerator,
                                          filterSizeGenerator,
                                          normalisation=filterNormalisation)
    importances2 = [55, 67, 81, 1, 19, 77, 21, 91, 57, 51, 27, 14, 89, 24, 23,
                    63, 94, 38, 98, 12, 9, 56, 31, 26, 79,  5,  8, 72, 28, 66,
                    59, 52, 71, 76, 80, 32, 6, 90, 18, 92, 83, 11, 22, 86, 35,
                    96, 36, 42, 53, 4, 65, 43, 30, 88, 48, 29, 69, 20, 17, 85,
                    64, 46, 73, 13, 62, 84, 97, 0, 68, 95, 45, 70, 61, 33, 2,
                    10, 39, 78, 44, 34, 47, 75, 87, 82, 54, 40, 7, 25, 15, 74,
                    3, 58, 37, 49, 60, 41, 16, 50, 93, 99]

    importances = [0, 56,  2, 68, 82, 20, 22, 78, 92, 52, 28, 15, 58, 25, 24,
                   90, 95, 99, 13, 64, 39, 29, 57, 10, 27, 80, 32, 73,  7,  6,
                   9, 94, 93, 65, 63, 44, 53, 72, 48, 14, 11, 67, 97, 37, 60,
                   91, 33, 84, 36,  3, 76, 46, 77, 55, 87, 83, 19,  4, 40, 98,
                   70, 62, 16, 47, 88, 18, 71,  1, 85, 86, 66, 38, 23, 50, 79,
                   43, 35, 41, 59,  8, 21, 74, 96,  5, 49, 26, 51, 61, 34, 31,
                   75, 12, 89, 30, 69, 81, 100, 17, 54, 45, 42]

    importances = importances[1:]
    importances = [x-1 for x in importances]
    filterGenerator = Finite3SameFilter(baseFilterGenerator, nbFilters)
    filters = [filt for filt, _, _ in filterGenerator]

    for i in range(len(filters)):
        print filters[importances[i]].shape

    print histoFilters(filters)

    mainFilters = []
    for i in range(nbMainFilters):
        mainFilters.append(filters[importances[i]])

    print "=============="
    print histoFilters(mainFilters)

    ni = 1./9
#    mainFilters[printFilter] = np.array([[ni, ni, ni], [ni, ni, ni], [ni, ni, ni]])

    print mainFilters[printFilter]
    lena = getLena()
    lenaFiltered = applyFilter(lena, mainFilters[printFilter])
    lenaFreq, H, lenaFFreq = recap(lena, mainFilters[printFilter])
#    plotImage(lena)
#    plotImage(lenaFreq)
    plotImage(lenaFiltered)
    plotImage(H)
    plotImage(lenaFFreq)
#
