#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

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

class NumericalGradientNetwork:
  """Sequential neural network model that uses numerical differentiation.

  Parameters
  ----------
  modules : Module[]
    List of modules; used to grab trainable weights.
  h: float
    Step size.
  loss : Module
    Final output activation and loss function.
  """
  def __init__(self, modules, h, loss=None):
    self.modules = modules
    self.h = h
    self.loss = loss()

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

  def compute_grad(self, X, idx, y):
    """Compute gradient of module.

    Parameters
    ----------
    X : np.array
      Input data.
    idx : int
      Module index.
    y : np.array
      True labels.
    """
    module = self.modules[idx]
    W, b = module.trainable_parameters

    W.grad = np.empty_like(W.value)
    for r in range(W.value.shape[0]):
      for c in range(W.value.shape[1]):
        W.value[r, c] += self.h
        pred = self.forward(X)
        loss_plus = categorical_cross_entropy(pred, y)

        W.value[r, c] -= (2 * self.h)
        pred = self.forward(X)
        loss_minus = categorical_cross_entropy(pred, y)

        W.value[r, c] += self.h
        W.grad[r, c] = (loss_plus - loss_minus) / (2 * self.h)

    b.grad = np.empty_like(b.value)
    for c in range(b.value.shape[0]):
      b.value[c] += self.h
      pred = self.forward(X)
      loss_plus = categorical_cross_entropy(pred, y)
      
      b.value[c] -= (2 * self.h)
      pred = self.forward(X)
      loss_minus = categorical_cross_entropy(pred, y)

      b.value[c] += self.h
      b.grad[c] = (loss_plus - loss_minus) / (2 * self.h)

  def backward(self, X, y):
    """Model backwards pass.

    Parameters
    ----------
    X : np.array
      Input data.
    y : np.array
      True labels.
    """
    for idx in range(len(self.modules)):
      self.compute_grad(X, idx, y)

  def train(self, dataset):
    """Fit model on dataset for a single epoch.

    Parameters
    ----------
    dataset : Dataset
      Training dataset with batches already split.

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
      postfix={"loss": 0, "accuracy": 0}) as pbar:
      for i, batch in enumerate(dataset):
        X, y = batch
        pred = self.forward(X)
        self.backward(X, y)

        losses[i] = categorical_cross_entropy(pred, y)
        accuracy[i] = categorical_accuracy(pred, y)
        pbar.update(1)
        pbar.set_postfix(loss=losses[i], accuracy=accuracy[i])
    return np.mean(losses), np.mean(accuracy)
