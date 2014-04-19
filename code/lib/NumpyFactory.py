# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:09:00 2014

@author: Jm
"""

#import tempfile
import shutil
import os
import sys
import atexit
import numpy as np


def delete_folder(folder_path):
    """Utility function to cleanup a temporary folder if still existing"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


class NumpyFactory:
#Heavily based on joblib

    SYSTEM_SHARED_MEM_FS = '/dev/shm'
    counter = 0

    def __init__(self, tmpFolder=SYSTEM_SHARED_MEM_FS, autoClean=True):

        # Prepare a sub-folder name for the serialization of this particular
        # pool instance (do not create in advance to spare FS write access if
        # no array is to be dumped):
        temp_folder = os.path.abspath(os.path.expanduser(tmpFolder))
        temp_folder = os.path.join(temp_folder,
                                   "joblib2_memmaping_pool_%d_%d" % (
                                       os.getpid(), id(self)))
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        self._temp_folder = temp_folder
        self._files = {}
        if autoClean:
            atexit.register(lambda: delete_folder(temp_folder))

    def createArray(self, shape, dtype):
        # TODO : lock
        c = NumpyFactory.counter
        NumpyFactory.counter += 1

        filePath = os.path.join(self._temp_folder, "f"+str(c)+".dat")
        array = np.memmap(filePath, dtype=dtype, shape=shape, mode='w+')
        self._files[id(array)] = filePath
        return array

    def clean(self, array):
        if not self._files.has_key:
            return False
        # TODO XXX : manage exceptions
        try:
            filePath = self._files[id(array)]
            delete_folder(filePath)
            del self._files[filePath]
        except:
            print "Unexpected error:", sys.exc_info()[0]
            return False
        return True

    def cleanAll(self):
        delete_folder(self._temp_folder)

if __name__ == "__main__":
    npFact = NumpyFactory()
    X = npFact.createArray((50000, 50000))
