#!/usr/bin/env python

import numpy as np

from common import Module, Parameter

class Dense(Module):
  """NumPy implementation of the Dense Layer.

  Parameters
  ----------
  in_dim : int
    Length of input dimensions.
  out_dim : int
    Length of output dimensions.
  """
  def __init__(self, in_dim, out_dim):
    W = np.zeros(shape=(out_dim, in_dim))
    b = np.zeros(shape=(out_dim,))
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
