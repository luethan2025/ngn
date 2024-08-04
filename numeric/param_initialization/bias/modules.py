#!/usr/bin/env python

import numpy as np

from .base import BiasInitializer

class Zero(BiasInitializer):
  def __init__(self, out_dim):
    """Zero initialization.

    Parameters
    ----------
    out_dim : int
      Length of output dimensions.
    """
    self.out_dim = out_dim

  def init_params(self):
    """Apply zero initialization.

    Returns
    -------
    np.array
      Initialized bias.
    """
    return np.zeros(shape=(self.out_dim))
