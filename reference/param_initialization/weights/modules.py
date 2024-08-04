#!/usr/bin/env python

import numpy as np

from .base import WeightInitializer

class Xavier(WeightInitializer):
  """Uniform Xavier initialization.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  """
  def __init__(self, in_dim, out_dim):
    self.in_dim = in_dim
    self.out_dim = out_dim

  def init_params(self):
    """Apply Uniform Xavier initialization.

    Returns
    -------
    np.array.
      Initialized weight matrix.
    """
    u = np.sqrt(6 / float(self.in_dim + self.out_dim))
    return np.random.uniform(low=-u, high=u, size=(self.out_dim, self.in_dim))

class He(WeightInitializer):
  """Uniform He initialization.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  """
  def __init__(self, in_dim, out_dim):
    self.in_dim = in_dim
    self.out_dim = out_dim

  def init_params(self):
    """Apply Uniform He initialization.

    Returns
    -------
    np.array.
      Initialized weight matrix.
    """
    u = np.sqrt(6 / float(self.in_dim))
    return np.random.uniform(low=-u, high=u, size=(self.out_dim, self.in_dim))
