#!/usr/bin/env python

class Parameter:
  """Container for a trainable parameter.
  
  Attributes
  ----------
  value: np.float64
    Parameter value.
  grad: np.float64
    The gradient of the parameter with respect to the loss function.
  """
  def __init__(self, value):
    self.value = value
    self.grad = None
