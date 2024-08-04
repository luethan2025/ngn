#!/usr/bin/env python

import numpy as np

class Dataset:
  """Dataset iterator.

  Parameters
  ----------
  X : np.array
    Input data points. Should have shape (dataset size, features).
  y : np.array
    Output one-hot labels. Should have shape (dataset size, classes).
  batch : int
    Number samples used in one forward and backward pass (defaults to 32).
  seed : int
    NumPy random seed used in benchmarking tests.
  """
  def __init__(self, X, y, batch=32, seed=None):
    self.X = X
    self.y = y
    self.batch = batch
    self.size = X.shape[0] // batch
    self.seed = seed

  def __iter__(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    self.idx = 0
    self.indices = np.random.permutation(
      self.X.shape[0]
    )[:self.size * self.batch].reshape(self.size, self.batch)
    return self
  
  def __next__(self):
    if self.idx < self.size:
      batch = self.indices[self.idx]
      self.idx += 1
      return (self.X[batch], self.y[batch])
    else:
      raise StopIteration()
