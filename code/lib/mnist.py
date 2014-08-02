# -*- coding: utf-8 -*-
"""
Created on Sat Aug 02 14:34:40 2014

@author: Jm
"""

import cPickle
import gzip
import numpy as np
import pylab as pl


def loadMnist(filename="mnist.pkl.gz"):
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_set_imgs = []
    train_set_labels = np.hstack((train_set[1], valid_set[1]))

    for img in train_set[0]:
        train_set_imgs.append(img.reshape(28, 28))

    for img in valid_set[0]:
        train_set_imgs.append(img.reshape(28, 28))

    test_set_imgs = []
    test_set_labels = test_set[1]
    for img in test_set[0]:
        test_set_imgs.append(img.reshape(28, 28))

    return train_set_imgs, train_set_labels, test_set_imgs, test_set_labels




if __name__ == "__main__":

    train_set_imgs, train_set_labels, test_set_imgs, test_set_labels = loadMnist()
    print len(train_set_imgs)
    print len(train_set_labels)
    print len(test_set_imgs)
    print len(test_set_labels)

    index = np.random.randint(0, len(train_set_imgs))
    pl.imshow(train_set_imgs[index])
    titl = "label "+str(train_set_labels[index])
    print titl
#    pl.title("test")
