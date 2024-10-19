#!/usr/bin/env python

class SGD:
  """Stochastic Gradient Descent (SGD) optimizer.

  Parameters
  ----------
  params : Parameter[]
    List of parameters that the gradients correspond to.
  lr : float
    Learning rate multiplier (defaults to 0.01).
  """
  def __init__(self, params, lr=0.01):
    self.params = params
    self.lr = lr

  def apply_gradients(self):
    """Apply gradients to parameters."""
    for p in self.params:
      p.value -= self.lr * p.grad
