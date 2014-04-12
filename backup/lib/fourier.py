# -*- coding: utf-8 -*-
"""
Created on Mon Apr 07 15:16:21 2014

@author: Jm
"""

import numpy as np
import PIL.Image as PImg
from scipy import signal as sg
import pylab as pl


def imshow(img, title=None, interpolation="none"):
    pl.figure()
    pl.imshow(img, cmap="gray", interpolation=interpolation)
    if title is not None:
        pl.title(title)


def combine(filt1, filt2):
    return sg.convolve2d(filt1, filt2, "same", "fill", 0)


def test(img, filt):
    imshow(img, "Original image")

    imgFreq = np.fft.fft2(img)
    imgFreqShow = np.array(np.log10(np.abs(np.fft.fftshift(imgFreq))+10),
                           dtype=np.uint8)
    imshow(imgFreqShow, "Original image in frequency domain (log scale)")

    filtFreq = np.fft.fft2(filt, img.shape)
    #filtShow = np.array(np.abs(np.fft.fftshift(filtFreq)), dtype=np.uint8)
    filtShow = np.abs(np.fft.fftshift(filtFreq))
    imshow(filtShow, "Filter in freqency domain")

    imgFF1 = imgFreq*filtFreq
#    imgFF1show = np.array(np.log10(np.abs(np.fft.fftshift(imgFF1))+10),
#                          dtype=np.uint8)
    imgFF1show = np.log10(np.abs(np.fft.fftshift(imgFF1))+10)

    imgF1 = np.fft.ifft2(imgFF1)
#    imgF1show = np.asarray(imgF1, dtype=np.uint8)
    imgF1show = np.abs(imgF1)

    imgF2 = sg.convolve2d(img, filt, "same", "fill", 0)
#    imgF2show = np.asarray(imgF2, dtype=np.uint8)
    imgF2show = np.abs(imgF2)

    imgFF2 = np.fft.fft2(imgF2)
#    imgFF2show = np.array(np.log10(np.abs(np.fft.fftshift(imgFF2))+10),
#                          dtype=np.uint8)
    imgFF2show = np.log10(np.abs(np.fft.fftshift(imgFF2))+10)

    imshow(imgF1show, "Filtered image (manually)")
    imshow(imgFF1show,
           "Filtered image (manually) in frequency domain (log scale)")

    imshow(imgF2show, "Filtered image (scipy)")
    imshow(imgFF2show,
           "Filtered image (scipy) in frequency domain (log scale)")

    return imgFreq, filtFreq, imgF2, imgFF2


if __name__ == "__main__":
    img = np.asarray(PImg.open("lena.png"))[:, :, 0]
#    img = np.asarray(PImg.open("frog.jpg"))[:, :, 0]

    #HighPass
    filt = np.array([[1., -2.,  1.], [-2.,  5., -2.], [1., -2.,  1.]])

    #Mean
    avg = np.array([[1./9, 1./9,  1./9], [1./9,  1./9, 1./9],
                    [1./9, 1./9,  1./9]])

    #HigPass followed by avg
    filt = combine(filt, avg)
    print filt

    #Filter0
#    filt = np.array([[1.30361606, -1.24722408, -0.8898184],
#                    [-0.28962705,  1.24625125,  1.44969196],
#                    [-0.02373095, -0.85853783, -0.69062096]])

    #Filter1
#    filt = np.array([[1.12956286, -0.51074656,  0.84887965],
#                    [0.69268987,  0.97054394, -1.07020246],
#                    [-0.6741272,   0.48137937, -1.86797948]])

    #Filter2
#    filt = np.array([[1.17455332,  0.26911437, -1.73740435],
#                     [0.5735455,  -0.07646283,  1.62503762],
#                     [-1.09551086, -0.57281528, -0.16005748]])

    imgFreq, filtFreq, imgF2, imgFF2 = test(img, filt)
