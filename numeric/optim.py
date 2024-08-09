#!/usr/bin/env python

class SGD:
  """Stochastic Gradient Descent (SGD) optimizer.

  Parameters
  ----------
  lr : float
    Learning rate multiplier (defaults to 0.01).
  """
  def __init__(self, lr=0.01):
    self.lr = lr

  def apply_gradients(self, params):
    """Apply gradients to parameters.

    Parameters
    ----------
    params : Parameter[]
      List of parameters that the gradients correspond to.
    """
    for p in params:
      p.value -= self.lr * p.grad
