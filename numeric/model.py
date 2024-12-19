#!/usr/bin/env python

import numpy as np

from common.loss import categorical_cross_entropy
from common.optim import SGD

class Model:
  """Numeric gradient-based sequential neural network model.

  Parameters
  ----------
  modules : Module[]
    List of modules; used to grab trainable weights.
  loss : Module
    Final output activation and loss function.
  """
  def __init__(self, modules, loss=None):
    self.modules = modules
    self.loss = loss()

    self.params = []
    for module in modules:
      self.params += module.trainable_parameters
    self.optim = SGD(self.params)
    self.X = None

  def forward(self, X):
    """Model forward pass.

    Parameters
    ----------
    X : np.array
      Input data.

    Returns
    -------
    np.array
      Batch predictions; should have shape (batch, num_classes).
    """
    self.X = X
    for module in self.modules:
      X = module.forward(X)
    return self.loss.forward(X)

  def _compute_grad(self, param, y, h=1e-6):
    """Compute the gradient w.r.t to the model's trainable parameters.
  
    Parameters
    ----------
    param : Parameter
      Trainable parameter.
    y : nparray
      True labels.
    h : float
      Step size.
    
    Return
    ------
    np.array
      Gradient w.r.t trainable parameter.
    """
    grad = np.empty_like(param.value)
    if param.value.ndim == 2:
      num_row, num_col = param.value.shape
      for r in range(num_row):
        for c in range(num_col):
          param.value[r, c] += h
          pred = self.forward(self.X)
          loss_plus = categorical_cross_entropy(pred, y)

          param.value[r, c] -= (2 * h)
          pred = self.forward(self.X)
          loss_minus = categorical_cross_entropy(pred, y)

          param.value[r, c] += h
          grad[r, c] = (loss_plus - loss_minus) / (2 * h)
    elif param.value.ndim == 1:
      num_col = param.value.shape
      for c in range(param.value.shape[0]):
        param.value[c] += h
        pred = self.forward(self.X)
        loss_plus = categorical_cross_entropy(pred, y)
        
        param.value[c] -= (2 * h)
        pred = self.forward(self.X)
        loss_minus = categorical_cross_entropy(pred, y)

        param.value[c] += h
        grad[c] = (loss_plus - loss_minus) / (2 * h)
    return grad

  def backward(self, y):
    """Model backwards pass.

    Parameters
    ----------
    y : np.array
      True labels.
    """
    for param in self.params:
      param.grad = self._compute_grad(param, y)
