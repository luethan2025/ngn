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
  """
  def __init__(self, in_dim, out_dim):
    u = np.sqrt(6 / (in_dim + out_dim))
    W = np.random.uniform(low=-u, high=u, size=(out_dim, in_dim))
    b = np.zeros(shape=(out_dim))

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

  def backward(self, grad):
    """Backward propagation for Dense.

    Parameters
    ----------
    grad : np.array
      Gradient (Loss w.r.t. data) flowing backwards from the next layer. Should
      have dimensions (batch, dim).
  
    Returns
    -------
    np.array
      Gradients for the inputs to this module. Should have dimensions
      (batch, dim).
    """
    W, b = self.trainable_parameters
    batch = self.x.shape[0]
    W.grad = np.matmul(grad.T, self.x) / batch
    b.grad = np.sum(grad, axis=0) / batch
    dx = np.matmul(grad, W.value)
    return dx
