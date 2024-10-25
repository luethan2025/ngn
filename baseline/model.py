#!/usr/bin/env python

from common.optim import SGD

class Model:
  """Reference sequential neural network model.

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

    params = []
    for module in modules:
      params += module.trainable_parameters
    self.optim = SGD(params)

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
    for module in self.modules:
      X = module.forward(X)
    return self.loss.forward(X)

  def backward(self, y):
    """Model backwards pass.

    Parameters
    ----------
    y : np.array
      True labels.
    """
    grad = self.loss.backward(y)
    for module in reversed(self.modules):
      grad = module.backward(grad)
