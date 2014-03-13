# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 08 2014
"""
Several tools for parallel computation
"""
import copy_reg
import types

import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from Coordinator import Coordinator

__all__ = ["TaskManager", "ParallelCoordinator"]


class TaskManager:
    """
    ===========
    TaskManager
    ===========
    A toolkit for preprocessing parallel computation
    """
    def computePartition(self, nbTasks, dataSize):
        """
        Compute data partitioning for parallel computation.

        Parameters
        ----------
        nbTasks : int {-1, >0}
            The parallelization factor. If -1 : the greatest factor is chosen
        dataSize : int > 0
            The size of the data to process

        Return
        ------
        triplet = (nbTasks, counts, starts)
        nbTasks : int
            The final parallelization factor. It is computed as
            min(#cpu/nbTasks, dataSize)
        counts : list of int
            The number of data pieces for each parallel task
        starts : list of int
            The start indexes of the data for each parallel task
        """
        if nbTasks == -1:
            nbTasks = min(cpu_count(), dataSize)
        else:
            nbTasks = min(nbTasks, dataSize)

        counts = [dataSize / nbTasks] * nbTasks

        for i in xrange(dataSize % nbTasks):
            counts[i] += 1

        starts = [0] * (nbTasks + 1)

        for i in xrange(1, nbTasks + 1):
            starts[i] = starts[i - 1] + counts[i - 1]

        return nbTasks, counts, starts

    def partition(self, nbTasks, data):
        """
        Partition the data for parallel computation.

        Parameters
        ----------
        nbTasks : int {-1, >0}
            The parallelization factor. If -1 : the greatest factor is chosen
        data : list
            The data to partition

        Return
        ------
        pair = (nbTasks, dataParts)
        nbTasks : int
            The final parallelization factor. It is computed as
            min(#cpu/nbTasks, dataSize)
        dataParts : list of lists
            each element of dataParts is a contiguous sublist of data : the
            partition for a parallel computation unit
        """
        nbTasks, counts, starts = self.computePartition(nbTasks, len(data))
        dataParts = []
        for i in xrange(nbTasks):
            dataParts.append(data[starts[i]:starts[i + 1]])
        return nbTasks, dataParts


def reduceMethod(m):
    """Adds the capacity to pickle method of objects"""
    return (getattr, (m.__self__, m.__func__.__name__))


class ParallelCoordinator(Coordinator):
    """
    ===================
    ParallelCoordinator
    ===================
    A coordinator (see :class:`Coordinator`) for parallel computing
    """
    #Counts the number of instances already created
    instanceCounter = 0

    def __init__(self, coordinator, nbParal=-1, verbosity=0):
        """
        Construct a :class:`ParallelCoordinator`

        Parameters
        ----------
        coordinator : :class:`Coordinator`
            The coordinator which will execute the work in its private
            child process
        nbParal : int {-1, > 0} (default : -1)
            The parallel factor. If -1, or > #cpu, the maximum factor is used
            (#cpu)
        verbosity : int >=0 (default : 0)
            The verbosity level. The higher, the more information message.
            Information message are printed on the stderr
        """
        self._coordinator = coordinator
        self._nbParal = nbParal
        self._verbosity = verbosity

        if ParallelCoordinator.instanceCounter == 0:
            copy_reg.pickle(types.MethodType, reduceMethod)
        ParallelCoordinator.instanceCounter += 1

    def process(self, imageBuffer):
        taskManager = TaskManager()
        nbJobs, subImageBuffer = taskManager.partition(self._nbParal,
                                                       imageBuffer)

        allData = Parallel(n_jobs=nbJobs, verbose=self._verbosity)(
            delayed(self._coordinator.process)(
                subImageBuffer[i])
            for i in xrange(nbJobs))

        # Reduce
        X = np.vstack(X for X, _ in allData)
        y = np.concatenate([y for _, y in allData])

        return X, y
        
if __name__ == "__main__":
    test1 = [("A",1), ("B",2), ("C",3), ("D",4)]
    test2 = [("A",1),("B",2),("C",3),("D",4),("E",5),("F",6),("G",7),("H",8)]
    test3 = [("A",1),("B",2),("C",3),("D",4),("E",5),("F",6),("G",7),("H",8),("I",9),("J",10),("K",11),("L",12)]
    
    taskMan = TaskManager()

