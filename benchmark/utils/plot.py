#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def plot_loss_and_acc(losses, accs, labels=None, figsize=(15,7)):
  assert len(losses) == len(accs)
  fig, axarr = plt.subplots(1, 2, figsize=figsize)
  axarr = np.array(axarr).reshape(-1) 

  ax = axarr[0]
  for idx, loss in enumerate(losses):
    ax.plot(loss, label=labels[idx] if labels else None)
  if labels:
    ax.legend()
  ax.set_title("Loss over Epochs")
  ax.set_xticks(range(len(loss)))

  ax = axarr[1]
  for idx, acc in enumerate(accs):
    ax.plot(acc, label=labels[idx] if labels else None)
  if labels:
    ax.legend()
  ax.set_title("Accuracy over Epochs")
  ax.set_xticks(range(len(acc)))
  plt.tight_layout()
  plt.show()
