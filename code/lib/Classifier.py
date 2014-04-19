# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
""" """
import numpy as np

from sklearn.metrics import confusion_matrix

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
        self._coord.clean(X, y)

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

        X_pred, _ = self._coord.process(image_buffer, learningPhase=False)

        self.endTask()
        return self._predict_proba(X_pred, len(image_buffer))

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
