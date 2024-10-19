#!/usr/bin/env python

import numpy as np

def categorical_accuracy(pred, labels):
  """Accuracy statistic.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).

  Returns
  -------
  float
    Mean accuracy in this batch.
  """
  return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))
