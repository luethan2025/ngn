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
  batch_size : int
    Number samples used in one forward and backward pass (defaults to 32).
  """
  def __init__(self, X, y, batch_size=32):
    self.X = X
    self.y = y
    self.batch_size = batch_size
    self.size = X.shape[0] // batch_size

  def __iter__(self):
    self.idx = 0
    self.indices = np.random.permutation(
      self.X.shape[0]
    )[:self.size * self.batch_size].reshape(self.size, self.batch_size)
    return self
  
  def __next__(self):
    if self.idx < self.size:
      batch = self.indices[self.idx]
      self.idx += 1
      return (self.X[batch], self.y[batch])
    else:
      raise StopIteration()
