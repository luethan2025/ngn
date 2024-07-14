#!/usr/bin/env python

import numpy as np

from .base import Module, Parameter

class Dense(Module):
  """NumPy implementation of the Dense Layer.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  initialization_technique : Techique
    Weight and bias initialization technique.
  """
  def __init__(self, in_dim, out_dim, initialization_technique=None):
    W, b = initialization_technique(in_dim, out_dim).initialize_weights_and_bias()
    self.trainable_parameters = [Parameter(W), Parameter(b)]

  def forward(self, x):
    """Forward propagation through Dense.

    Parameters
    ----------
    x : np.array
      Input for this layer.
  
    Returns
    -------
    np.array
      Output of this layer.
    """
    self.x = x
    W, b = self.trainable_parameters
    return np.tensordot(W.value, x, axes=[1, 1]).T + b.value
