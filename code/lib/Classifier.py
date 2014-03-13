# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
""" """
import numpy as np

__all__ = ["Classifier"]


class Classifier:
    """
    ==========
    Classifier
    ==========
    A :class:`Classifier` uses a :class:`Coordinator` to extract data from
    an :class:`ImageBuffer` and feed it to a **scikit-learn base classifier**.
    The :class:`Classifier` can take care of multiple feature vectors per
    object.

    Note
    ----
    Internally, the labels will go from 0 to N-1, where N correspond to the
    number of different input labels. The order is preserved. The labels
    returned will follow this convention.
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
        self._classifier = base_classifier
        self._coord = coordinator

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
        X, y = self._coord.process(image_buffer)
        # Count the number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y = np.searchsorted(self.classes_, y)

        self._classifier.fit(X, y)

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
            of the same index as computed by the base classifier
        """
        return self.classes_.take(
            np.argmax(self.predict_proba(image_buffer), axis=1),  axis=0)

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
        X_pred, _ = self._coord.process(image_buffer)

        aggreg = len(X_pred)/len(image_buffer)

        y = np.zeros((len(image_buffer), self.n_classes_))

        _y = self._classifier.predict_proba(X_pred)

        for i in xrange(len(image_buffer)):
                y[i] = np.sum(_y[i * aggreg:(i + 1) * aggreg], axis=0) / aggreg

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
        return sum( map( (lambda x,y:x==y), y_pred, y_truth ) )/len(y_truth)
