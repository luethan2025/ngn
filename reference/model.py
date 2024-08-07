#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .optim import SGD

def categorical_cross_entropy(pred, labels, epsilon=1e-10):
  """Cross entropy loss function.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).
  epsilon : float
    Small constant to add to the log term of cross entropy to help with
    numerical stability (defaults to 1e-10).

  Returns
  -------
  float
    Mean cross entropy loss in this batch.
  """
  return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))

def categorical_accuracy(pred, labels):
  """Accuracy statistic.

  Parameters
  ----------
  pred : np.array
    Softmax label predictions. Should have shape (dim, num_classes).
  labels : np.array
    One-hot true labels. Should have shape (dim, num_classes).

  Returns
  -------
  float
    Mean accuracy in this batch.
  """
  return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))

class Model:
  """Reference sequential neural network model.

  Parameters
  ----------
  modules : Module[]
    List of modules; used to grab trainable weights.
  loss : Module
    Final output activation and loss function.
  optim : SGD
    Optimization policy (defaults to SGD).
  """
  def __init__(self, modules, loss=None, optim=SGD):
    self.modules = modules
    self.loss = loss()

    self.params = []
    for module in modules:
      self.params += module.trainable_parameters
    self.optim = optim()

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

  def train(self, dataset, silence=True):
    """Fit model on dataset for a single epoch.

    Parameters
    ----------
    dataset : Dataset
      Training dataset with batches already split.
    silence : boolean
      Silence tqdm (defaults to True).

    Returns
    -------
    (float, float)
      [0] Mean train loss during this epoch.
      [1] Mean train accuracy during this epoch.
    """
    losses = np.zeros(shape=dataset.size)
    accuracy = np.zeros(shape=dataset.size)
    with tqdm(
      total=dataset.size,
      postfix={"loss": 0, "accuracy": 0},
      disable=silence) as pbar:
      for i, batch in enumerate(dataset):
        X, y = batch
        pred = self.forward(X)
        self.backward(y)
        self.optim.apply_gradients(self.params)

        losses[i] = categorical_cross_entropy(pred, y)
        accuracy[i] = categorical_accuracy(pred, y)
        pbar.update(1)
        pbar.set_postfix(loss=losses[i], accuracy=accuracy[i])
    return np.mean(losses), np.mean(accuracy)
