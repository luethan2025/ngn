#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .evaluate import categorical_accuracy
from .loss import categorical_cross_entropy

def train_model(model, dataset, num_epoch, silence=True):
  """Trains a neural network for a set number of epochs.

  Parameters
  ----------
  model: Model
    Neural network.
  dataset : Dataset
    Training dataset with batches already split.
  num_epoch:
    Number of epochs to train for.
  silence : boolean
    Silence tqdm (defaults to True).

  Returns
  -------
  (np.array, np.array)
    [0] Mean train loss per epoch.
    [1] Mean train accuracy per epoch.
  """
  losses = np.zeros(num_epoch)
  accuracy = np.zeros(num_epoch)
  for epoch in range(num_epoch):
    curr_losses = np.zeros(shape=dataset.size)
    curr_accuracy = np.zeros(shape=dataset.size)
    with tqdm(
      total=dataset.size,
      postfix={"loss": 0, "accuracy": 0},
      disable=silence) as pbar:
        for i, batch in enumerate(dataset):
          X, y = batch
          pred = model.forward(X)
          model.backward(y)
          model.optim.apply_gradients()

          curr_losses[i] = categorical_cross_entropy(pred, y)
          curr_accuracy[i] = categorical_accuracy(pred, y)

          pbar.update(1)
          pbar.set_postfix(loss=curr_losses[i], accuracy=curr_accuracy[i])
    losses[epoch] = np.mean(curr_losses)
    accuracy[epoch] = np.mean(curr_accuracy)
  return losses, accuracy
