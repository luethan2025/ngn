#!/usr/bin/env python

import numpy as np

from .base import Module, Parameter
from ..param_initialization.weights import Xavier
from ..param_initialization.bias import Zero

class Dense(Module):
  """NumPy implementation of the Dense Layer.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  weight_initializer : WeightInitializer
    Weight initialization method (defaults to Xavier).
  bias_initializer : BiasInitializer
    bias initialization method (defaults to Zero).
  seed : int
    NumPy random seed used in benchmarking tests.
  """
  def __init__(
      self, in_dim, out_dim, weight_initializer=Xavier, bias_initializer=Zero, seed=None):
    if seed is not None:
      np.random.seed(seed)
    W = weight_initializer(in_dim, out_dim).init_params()
    b = bias_initializer(out_dim).init_params()
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

class SoftmaxCrossEntropy(Module):
  """Softmax Cross Entropy fused output activation."""
  def __init__(self):
    super().__init__()

  def forward(self, logits):
    """Forward propagation through Softmax.

    Parameters
    ----------
    logits : np.array
      Softmax logits. Should have shape (batch, num_classes).

    Returns
    -------
    np.array
      Predictions for this batch. Should have shape (batch, num_classes).
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    self.y_pred = np.divide(
        exp_logits, np.sum(exp_logits, axis=1, keepdims=True))
    return self.y_pred
