# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:46:12 2014

@author: Jm
"""
import numpy as np
from scipy import signal as sg
import pylab as pl


def imshow(img, title=None, interpolation="none"):
    pl.figure()
    pl.imshow(img, cmap="gray", interpolation=interpolation)
    pl.colorbar()
    if title is not None:
        pl.title(title)


def norm(img):
    return (abs(img)).sum()


if __name__ == "__main__":

    const = 127

    img = np.ones((32, 32), dtype=np.uint8)*const

    e1 = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype=np.float).reshape(3,3)
    e2 = np.ones((3,3))/9.


    j1 = sg.convolve2d(img, e1, "same", "fill", const)
    j2 = sg.convolve2d(img, e2, "same", "fill", const)

    imshow(img, "Constant image")
    imshow(j1, "J1 = I * e1")
    imshow(j2, "J2 = I * e2")

    d1 = img - j1
    d2 = img - j2

    imshow(d1, "img - j1")
    imshow(d2, "img - j2")

    print "Norm e1", norm(e1)
    print "Norm e2", norm(e2)
    print "-----------------"
    print "Norm j1", norm(j1)
    print "Norm j2", norm(j2)