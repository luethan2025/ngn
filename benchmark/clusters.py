#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

from common.dataset import Dataset
from common import train_model, test_model

from utils import plot_loss_and_acc

def get_argparser():
  parser = argparse.ArgumentParser()

  # Datset Options
  parser.add_argument("--num_points", type=int, default=1000,
                      help="num points (default: 1000)")
  parser.add_argument("--split_ratio", type=float, default=0.5,
                      help="split ratio (default: 0.5)")
  parser.add_argument("--centroid_one_x", type=float, default=0.75,
                      help="centroid one x (default: 0.75)")
  parser.add_argument("--centroid_one_y", type=float, default=0.75,
                      help="centroid one y (default: 0.75)")
  parser.add_argument("--centroid_one_spread_factor", type=float, default=0.75,
                      help="centroid one spread factor (default: 0.75)")
  parser.add_argument("--centroid_two_x", type=float, default=-0.75,
                      help="centroid one x (default: -0.75)")
  parser.add_argument("--centroid_two_y", type=float, default=-0.75,
                      help="centroid one y (default: -0.75)")
  parser.add_argument("--centroid_two_spread_factor", type=float, default=0.75,
                      help="centroid two spread factor (default: 0.75)")
  
  # Train Options
  parser.add_argument("--batch_size", type=int, default=32,
                      help="batch (default: 32)")
  parser.add_argument("--num_epoch", type=int, default=5,
                      help="num epoch (default: 5)")
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
                centroid_one_x, centroid_one_y, centroid_one_spread_factor, \
                centroid_two_x, centroid_two_y, centroid_two_spread_factor, \
                batch_size, visualize=False):
  num_class_one = int(num_points * split_ratio)
  angles_one = np.random.uniform(0, 2 * np.pi, num_class_one)
  radii_one = np.sqrt(np.random.uniform(0, centroid_one_spread_factor**2, num_class_one))
  x_one = centroid_one_x + radii_one * np.cos(angles_one)
  y_one = centroid_one_y + radii_one * np.sin(angles_one)
  class_one = np.column_stack((x_one, y_one))

  num_class_two = int(num_points - (num_points * split_ratio))
  angles_two = np.random.uniform(0, 2 * np.pi, num_class_two)
  radii_two = np.sqrt(np.random.uniform(0, centroid_two_spread_factor**2, num_class_two))
  x_two = centroid_two_x + radii_two * np.cos(angles_two)
  y_two = centroid_two_y + radii_two * np.sin(angles_two)
  class_two = np.column_stack((x_two, y_two))

  X = np.vstack((class_one, class_two))
  label_one = np.tile([1, 0], (num_class_one, 1))
  label_two = np.tile([0, 1], (num_class_two, 1))
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
  np.random.seed(opts.random_seed)
  random.seed(opts.random_seed)

  train_dataset = get_dataset(opts.num_points, opts.split_ratio, \
                              opts.centroid_one_x, opts.centroid_one_y, opts.centroid_one_spread_factor, \
                              opts.centroid_two_x, opts.centroid_two_y, opts.centroid_two_spread_factor, \
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
                               opts.centroid_one_x, opts.centroid_one_y, opts.centroid_one_spread_factor, \
                               opts.centroid_two_x, opts.centroid_two_y, opts.centroid_two_spread_factor, \
                               opts.batch_size)
    loss_b, accuracy_b = test_model(bn, test_dataset)
    loss_ngn, accuracy_ngn = test_model(ngn, test_dataset)
    out += f"Baseline Model Loss:               {loss_b:.3f}   Baseline Model Accuracy:               {accuracy_b:.3f}\n"
    out += f"Numeric Gradient-based Model Loss: {loss_ngn:.3f}   Numeric Gradient-based Model Accuracy: {accuracy_ngn:.3f}\n\n"
  out = out.strip()
  print(out)

if __name__ == "__main__":
  main()
