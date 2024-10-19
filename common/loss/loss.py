#!/usr/bin/env python

import numpy as np

def categorical_cross_entropy(pred, labels, epsilon=1e-10):
  """Cross entropy loss function.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).
  epsilon : float
    Small constant to add to the log term of cross entropy to help with
    numerical stability (defaults to 1e-10).

  Returns
  -------
  float
    Mean cross entropy loss in this batch.
  """
  return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))
