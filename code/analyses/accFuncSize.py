# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:40:09 2014

@author: Jm Begon
"""
import pylab as pl


def plotAccSize():
    sizes = [500, 5000, 10000, 20000, 30000, 40000, 50000]
    sizeLabels = ["500", "5000", "10000", "20000", "30000", "40000", "50000"]

    pixit = [0.312, 0.4272, 0.4398, 0.4704, 0.4815, 0.4903, 0.4977]
    rci_agg = [0.2517, 0.3161, 0.3398, 0.349, 0.3716, 0.3799, 0.3798]
    rci_mw = [0.3404, 0.4283, 0.4527, 0.4808, 0.4998, 0.5044, 0.5151]

    legend = ["rci : moving window", "rci : aggregation",
              "pixit"]
    pl.figure()
    pl.xticks(sizes, sizeLabels, rotation=-20)
    pl.xlabel("learning set size")
    pl.xlim(50, 50500)
    pl.yticks(pl.arange(0, 0.6, 0.05))
    pl.ylim(0.2, 0.6)
    pl.ylabel("Accuracy")
    #pl.plot(sizes, data.transpose(), "-o")
    pl.plot(sizes, rci_mw, "-o")
    pl.plot(sizes, rci_agg, ":o")
    pl.plot(sizes, pixit, "--o")
    pl.legend(legend)
    pl.title("Accuracy as a function of the learning set size")
    #pl.plot()
