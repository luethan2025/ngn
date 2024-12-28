#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

from common.dataset import Dataset
from common import train_model, test_model

from util import plot_loss_and_acc

def get_argparser():
  parser = argparse.ArgumentParser()

  # Datset Options
  parser.add_argument("--num_points", type=int, default=1000,
                      help="num points (default: 1000)")
  parser.add_argument("--split_ratio", type=float, default=0.5,
                      help="split ratio (default: 0.5)")
  parser.add_argument("--quadrant_one_split_ratio", type=float, default=0.5,
                      help="quadrant one split ratio; when added with quadrant_three_split_ratio the sum must equal 1.0 (default: 0.5)")
  parser.add_argument("--quadrant_two_split_ratio", type=float, default=0.5,
                      help="quadrant two split ratio; when added with quadrant_four_split_ratio the sum must equal 1.0 (default: 0.5)")
  parser.add_argument("--quadrant_three_split_ratio", type=float, default=0.5,
                      help="quadrant three split ratio; when added with quadrant_one_split_ratio the sum must equal 1.0 (default: 0.5)")
  parser.add_argument("--quadrant_four_split_ratio", type=float, default=0.5,
                      help="quadrant four split ratio; when added with quadrant_two_split_ratio the sum must equal 1.0 (default: 0.5)")
  
  # Train Options
  parser.add_argument("--batch_size", type=int, default=32,
                      help="batch size (default: 32)")
  parser.add_argument("--num_epoch", type=int, default=50,
                      help="num epoch (default: 50)")
  parser.add_argument("--random_seed", type=int, default=1,
                      help="random seed (default: 1)")
  
  # Test Options
  parser.add_argument("--num_test_dataset", type=int, default=5,
                      help="num test datasets (default: 5)")
  
  # Visualizer Options
  parser.add_argument("--visualize_dataset", action="store_true", default=False,
                      help="visualize dataset (default: False)")
  parser.add_argument("--visualize_loss_and_acc", action="store_true", default=False,
                      help="visualize dataset (default: False)")
  return parser

def get_dataset(num_points, split_ratio, \
                quadrant_one_split_ratio, quadrant_two_split_ratio, \
                quadrant_three_split_ratio, quadrant_four_split_ratio, \
                batch_size, visualize=False):
  num_class_one = int(num_points * split_ratio)
  x_one = np.random.uniform(0, 2, int(num_class_one * quadrant_one_split_ratio))
  x_one = np.concatenate([x_one, np.random.uniform(-2, 0, int(num_class_one * quadrant_four_split_ratio))])
  y_one = np.random.uniform(0, 2, int(num_class_one * quadrant_one_split_ratio))
  y_one = np.concatenate([y_one, np.random.uniform(-2, 0, int(num_class_one * quadrant_four_split_ratio))])
  class_one = np.column_stack((x_one, y_one))

  num_class_two = int(num_points - (num_points * split_ratio))
  x_two = np.random.uniform(-2, 0, int(num_class_two * quadrant_two_split_ratio))
  x_two = np.concatenate([x_two, np.random.uniform(0, 2, int(num_class_two * quadrant_three_split_ratio))])
  y_two = np.random.uniform(0, 2, int(num_class_two * quadrant_two_split_ratio))
  y_two = np.concatenate([y_two, np.random.uniform(-2, 0, int(num_class_two * quadrant_three_split_ratio))])
  class_two = np.column_stack((x_two, y_two))

  X = np.vstack((class_one, class_two))
  label_one = np.tile([1, 0], (int(num_class_one * quadrant_one_split_ratio), 1))
  label_one = np.concatenate([label_one, np.tile([1, 0], (int(num_class_one * quadrant_four_split_ratio), 1))])
  label_two = np.tile([0, 1], (int(num_class_two * quadrant_two_split_ratio), 1))
  label_two = np.concatenate([label_two, np.tile([0, 1], (int(num_class_two * quadrant_three_split_ratio), 1))])
  y = np.row_stack((label_one, label_two))

  if visualize:
    plt.scatter(x_one, y_one)
    plt.scatter(x_two, y_two)
    plt.xticks(np.arange(-2, 3, step=1))
    plt.yticks(np.arange(-2, 3, step=1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
  return Dataset(X, y, batch_size=batch_size)

def get_baseline_model():
  from baseline import Model, Dense, SoftmaxCrossEntropy
  bn = Model([
    Dense(in_dim=2, out_dim=2)
  ], loss=SoftmaxCrossEntropy)
  del Model, Dense, SoftmaxCrossEntropy
  return bn

def get_numeric_gradient_model():
  from numeric import Model, Dense, SoftmaxCrossEntropy
  ngn = Model([
    Dense(in_dim=2, out_dim=2)
  ], loss=SoftmaxCrossEntropy)
  del Model, Dense, SoftmaxCrossEntropy
  return ngn

def main():
  opts = get_argparser().parse_args()
  assert (opts.quadrant_one_split_ratio + opts.quadrant_three_split_ratio == 1.0)
  assert (opts.quadrant_two_split_ratio + opts.quadrant_four_split_ratio == 1.0)
  np.random.seed(opts.random_seed)
  random.seed(opts.random_seed)

  train_dataset = get_dataset(opts.num_points, opts.split_ratio, \
                              opts.quadrant_one_split_ratio, opts.quadrant_two_split_ratio, \
                              opts.quadrant_three_split_ratio, opts.quadrant_four_split_ratio, \
                              opts.batch_size, visualize=opts.visualize_dataset)
  bn = get_baseline_model()
  np.random.seed(opts.random_seed)
  losses_b, accuracy_b = train_model(bn, train_dataset, opts.num_epoch)

  np.random.seed(opts.random_seed)
  ngn = get_numeric_gradient_model()
  losses_ngn, accuracy_ngn = train_model(ngn, train_dataset, opts.num_epoch)

  if opts.visualize_loss_and_acc:
    plot_loss_and_acc([losses_b, losses_ngn],
                      [accuracy_b, accuracy_ngn], labels=["Baseline Model", "Numeric Gradient-based Model"])

  out = ""
  for i in range(opts.num_test_dataset):
    np.random.seed(opts.random_seed + i + 1)
    out += f"Evaluating on test dataset {i + 1}:\n"
    test_dataset = get_dataset(opts.num_points, opts.split_ratio, \
                              opts.quadrant_one_split_ratio, opts.quadrant_two_split_ratio, \
                              opts.quadrant_three_split_ratio, opts.quadrant_four_split_ratio, \
                              opts.batch_size)
    loss_b, accuracy_b = test_model(bn, test_dataset)
    loss_ngn, accuracy_ngn = test_model(ngn, test_dataset)
    out += f"Baseline Model Loss:               {loss_b:.3f}   Baseline Model Accuracy:               {accuracy_b:.3f}\n"
    out += f"Numeric Gradient-based Model Loss: {loss_ngn:.3f}   Numeric Gradient-based Model Accuracy: {accuracy_ngn:.3f}\n\n"
  out = out.strip()
  print(out)

if __name__ == "__main__":
  main()
