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

class Module:
  """Base class for network layers and activation functions.

  Attributes
  ----------
  self.trainable_parameters : Parameter[]
    List of parameters that can be trained in this module.
  """
  def __init__(self):
    self.trainable_parameters = []

  def forward(self, x):
    """Forward propagation.

    Parameters
    ----------
    x : np.array
      Input for this module.

    Returns
    -------
    np.array
      Output of this module.
    """
    raise NotImplementedError()
  
  def backward(self, grad):
    """Backward propagation.

    Parameters
    ----------
    grad : np.array
      Gradient flowing backwards from the next module.

    Returns
    -------
    np.array
      Gradients for the inputs to this module.
    """
    raise NotImplementedError()
