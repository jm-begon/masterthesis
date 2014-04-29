# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
""" """
import numpy as np
import scipy.sparse as sps

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomTreesEmbedding

from Logger import Progressable


__all__ = ["Classifier"]


class Classifier(Progressable):
    """
    ==========
    Classifier
    ==========
    A :class:`Classifier` uses a :class:`Coordinator` to extract data from
    an :class:`ImageBuffer` and feed it to a **scikit-learn base classifier**.
    The :class:`Classifier` can take care of multiple feature vectors per
    object.
    """
    def __init__(self, coordinator, base_classifier):
        """
        Construct a :class:`Classifier`

        Parameters
        ----------
        coordinator : :class:`Coordinator`
            The coordinator responsible for the features extraction
        base_classifier : scikit-learn classifier (:meth:`predict_proba`
        required)
            The learning algorithm which will classify the data
        """
        Progressable.__init__(self, coordinator.getLogger())
        self._classifier = base_classifier
        self._coord = coordinator
        self._classifToUserLUT = []
        self._userToClassifLUT = {}

    def _buildLUT(self, y_user):
#        """
#        Builds the lookup tables for converting user labels to/from
#        classifier label
#
#        Parameters
#        ----------
#        y_user : list
#            the list of user labels
#        """
        userLabels = np.unique(y_user)
        self._classifToUserLUT = userLabels
        self._userToClassifLUT = {j: i for i, j in enumerate(userLabels)}

    def _convertLabel(self, y_user):
#        """
#        Convert labels from the user labels to the internal labels
#        Parameters
#        ----------
#        y_user : list
#            list of user labels to convert into internal labels
#        Returns
#        -------
#        y_classif : list
#            the corresponding internal labels
#        """
        return [self._userToClassifLUT[x] for x in y_user]

    def _convertLabelsBackToUser(self, y_classif):
#        """
#        Convert labels back to the user labels
#        Parameters
#        ----------
#        y_classif : list
#            list of internal labels to convert
#        Returns
#        -------
#        y_user : list
#            the corresponding user labels
#        """
        return [self._classifToUserLUT[x] for x in y_classif]

    def fit(self, image_buffer):
        """
        Fits the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to learn from

        Return
        -------
        self : :class:`Classifier`
            This instance
        """
        #Updating the labels
        y_user = image_buffer.getLabels()
        self._buildLUT(y_user)

        #Extracting the features
        self.setTask(1, "Extracting the features (model creation)")

        X, y_user = self._coord.process(image_buffer, learningPhase=True)

        self.endTask()

        #Converting the labels
        y = self._convertLabel(y_user)

        #Delegating the classification
        self.setTask(1, "Learning the model")

        self._classifier.fit(X, y)

        self.endTask()

        #Cleaning up
        self._coord.clean(X, y_user)

        return self

    def predict(self, image_buffer):
        """
        Classify the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of int
            each entry is the classification label corresponding to the input
        """
        y_classif = np.argmax(self.predict_proba(image_buffer), axis=1)
        return self._convertLabelsBackToUser(y_classif)

    def predict_proba(self, image_buffer):
        """
        Classify softly the data contained is the :class:`ImageBuffer`
        instance. i.e. yields a probability vector of belongin to each
        class

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of list of float
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        """
        #Extracting the features
        self.setTask(1, "Extracting the features (prediction)")

        X_pred, y = self._coord.process(image_buffer, learningPhase=False)

        #Cleaning up
        self._coord.clean(y)
        del y

        self.endTask()

        y = self._predict_proba(X_pred, len(image_buffer))

        #Cleaning up
        self._coord.clean(X_pred)
        del X_pred

        return y

    def _predict_proba(self, X_pred, nb_objects):
        #Misc.
        nbFactor = len(X_pred)/nb_objects

        y = np.zeros((nb_objects, len(self._userToClassifLUT)))

        #Classifying the data
        self.setTask(1, "Classifying (prediction)")

        _y = self._classifier.predict_proba(X_pred)

        self.endTask()

        for i in xrange(nb_objects):
                y[i] = np.sum(_y[i * nbFactor:(i + 1) * nbFactor], axis=0) / nbFactor

        return y

    def _predict(self, X_pred, nb_objects):
        y_classif = np.argmax(self._predict_proba(X_pred, nb_objects), axis=1)
        return self._convertLabelsBackToUser(y_classif)

    def accuracy(self, y_pred, y_truth):
        """
        Compute the frequency of correspondance between the two vectors

        Parameters
        -----------
        y_pred : list of int
            The prediction by the model
        y_truth : list of int
            The ground truth

        Return
        -------
        accuracy : float
            the accuracy
        """
        return sum(map((lambda x, y: x == y), y_pred, y_truth))/float(len(y_truth))

    def confusionMatrix(self, y_pred, y_truth):
        """
        Compute the confusion matrix

        Parameters
        -----------
        y_pred : list of int
            The prediction by the model
        y_truth : list of int
            The ground truth

        Return
        -------
        mat : 2D numpy array
            The confusion matrix
        """
        return confusion_matrix(y_truth, y_pred)


class UnsupervisedVisualBagClassifier(Classifier):
    """
    ===============================
    UnsupervisedVisualBagClassifier
    ===============================
    1. Unsupervised
    2. Binary bag of words
    3. Totally random trees
    """

    def __init__(self, coordinator, base_classifier, n_estimators=10,
                 max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 n_jobs=-1, random_state=None, verbose=0, min_density=None):
        Classifier.__init__(self, coordinator, base_classifier)
        self._visualBagger = RandomTreesEmbedding(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf,
                                                  n_jobs=n_jobs,
                                                  random_state=random_state,
                                                  verbose=verbose,
                                                  min_density=min_density)

    def _preprocess(self, image_buffer, learningPhase):
        if learningPhase:
            self.setTask(1, "Extracting the features (model creation)")
        else:
            self.setTask(1, "Extracting the features (prediction)")

        X_pred, y = self._coord.process(image_buffer,
                                        learningPhase=learningPhase)

        y_user = self._convertLabel(y)

        #Cleaning up
        self._coord.clean(y)
        del y

        self.endTask()

        #Bag-of-word transformation
        self.setTask(1, "Transforming data into bag-of-words")

        X2 = None
        if learningPhase:
            X2 = self._visualBagger.fit_transform(X_pred, y_user)
        else:
            X2 = self._visualBagger.transform(X_pred)

        #Cleaning up
        self._coord.clean(X_pred)
        del X_pred
        del y_user

        height = len(image_buffer)
        width = X2.shape[1]
        nbFactor = X2.shape[0] // height

        X3 = sps.csr_matrix((height, width), dtype=np.float64)

        self.logMsg("X3 shape : "+str(X3.shape), 35)
        self.logMsg("X3 dtype : "+str(X3.dtype), 35)
        #self.logSize("X3 total size : ", (X3.size*X3.itemsize), 35)

        rCoord, cCoord = X2.nonzero()
        for r, c in zip(rCoord, cCoord):
            X3[r//nbFactor, c] += 1

#        startIndex = 0
#        endIndex = startIndex + nbFactor
#        for row in xrange(height):
#            X3[row] = X2[startIndex:endIndex].sum(axis=0)
#            startIndex = endIndex
#            endIndex = startIndex + nbFactor

        #Cleaning up
        del X2  # Should be useless

        self.endTask()

        return X3

    def fit(self, image_buffer):
        """
        Fits the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to learn from

        Return
        -------
        self : :class:`Classifier`
            This instance
        """
        #Updating the labels
        y_user = image_buffer.getLabels()
        self._buildLUT(y_user)
        y = self._convertLabel(y_user)

        X = self._preprocess(image_buffer, learningPhase=True)

        #Delegating the classification
        self.setTask(1, "Learning the model")

        self._classifier.fit(X, y)

        self.endTask()

        return self

    def predict(self, image_buffer):
        """
        Classify the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of int
            each entry is the classification label corresponding to the input
        """

        X = self._preprocess(image_buffer, learningPhase=False)
        y_classif = self._classifier.predict(X)
        return self._convertLabelsBackToUser(y_classif)

    def predict_proba(self, image_buffer):
        """
        Classify softly the data contained is the :class:`ImageBuffer`
        instance. i.e. yields a probability vector of belongin to each
        class

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of list of float
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        """
        if not hasattr(self._classifier, "predict_proba"):
            #Early error
            self._classifier.predict_proba(np.zeros((1, 1)))

        X = self._preprocess(image_buffer, learningPhase=False)
        return self._classifier.predict_proba(X)
