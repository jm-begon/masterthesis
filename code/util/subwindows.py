# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:23:38 2014

@author: Jm
"""
import numpy as np

def totalSW(w, h):
    h2 = h**2
    w2 = w**2
    return (1./4) * (h2*w2 + h2*w + h*w2 + h*w)


def nbSW1D(S, m, M=None):
    """
    S :
        The size
    m :
        the minimum size
    M :
        the maximum size
    """
    if M is None:
        M = m
    res = 0
    for i in xrange(m, M+1):
        res += S - i + 1
    return res


def nbSW1D2(S, m, M=None):
    """
    S :
        The size
    m :
        the minimum size
    M :
        the maximum size
    """
    if M is None:
        M = m
    return int((M-m+1)*(S+1) - ((M+1)*M)/2. + ((m-1)*m)/2)


def nbSW(H, W, a, b, A=None, B=None):
    if A is None:
        A = a
    if B is None:
        B = b
    res = 0
    for i in xrange(a, A+1):
        for j in xrange(b, B+1):
            res += (H - i + 1) * (W - j + 1)
    return res


def nbSW2(H, W, a, b, A=None, B=None):
    if A is None:
        A = a
    if B is None:
        B = b
    return nbSW1D2(H, a, A)*nbSW1D2(W, b, B)


def nbSWonPixel(H, W, r, c):
    r = r+1
    c = c+1
    return (r*H - r**2 + r) * (c*W - c**2 + c)

def probPixels(H, W):
    prob = np.
