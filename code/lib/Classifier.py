# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
""" """
import numpy as np

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
        Fits the data contained is the :class:`ImageBuffer` instance

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
        self.logMsg("Extracting the features (model creation)...", 35)

        X, y_user = self._coord.process(image_buffer)

        self.logMsg("...Feature extraction done (model creation)", 45)

        #Converting the labels
        y = self._convertLabel(y_user)

        #Delegating the classification
        self.logMsg("Learning the model...", 35)

        self._classifier.fit(X, y)

        self.logMsg("...Model learnt", 45)

        return self

    def predict(self, image_buffer):
        """
        Classify the data contained is the :class:`ImageBuffer` instance

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
        list : list of list of int
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        """
        #Extracting the features
        self.logMsg("Extracting the features (prediction)...", 35)

        X_pred, _ = self._coord.process(image_buffer)

        self.logMsg("...Feature extraction done (prediction)", 45)

        #Misc.
        nbFactor = len(X_pred)/len(image_buffer)

        y = np.zeros((len(image_buffer), len(self._userToClassifLUT)))

        #Classifying the data
        self.logMsg("Classifying (prediction)...", 35)

        _y = self._classifier.predict_proba(X_pred)

        self.logMsg("...Classified (prediction)", 45)

        for i in xrange(len(image_buffer)):
                y[i] = np.sum(_y[i * nbFactor:(i + 1) * nbFactor], axis=0) / nbFactor

        return y

    def accuracy(self, y_pred, y_truth):
        """
        Computes the frequency of correspondance between the two vectors

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
