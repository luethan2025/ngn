#!/usr/bin/env python

from .evaluate import categorical_accuracy
from .loss import categorical_cross_entropy

def test_model(model, dataset):
  """Test a neural network.

  Parameters
  ----------
  model: Model
    Neural network.
  dataset : Dataset
    Training dataset with batches already split.

  Returns
  -------
  (float, float)
    [0] Mean test loss.
    [1] Test accuracy.
  """
  pred = model.forward(dataset.X)
  loss = categorical_cross_entropy(pred, dataset.y)
  acc = categorical_accuracy(pred, dataset.y)
  return loss, acc
