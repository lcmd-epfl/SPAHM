#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_kernel

parser = argparse.ArgumentParser(description='This program computes the learning curve.')
parser.add_argument('--x',      type=str,   dest='repr',      required=True, help='path to the representations file')
parser.add_argument('--y',      type=str,   dest='prop',      required=True, help='path to the properties file')
parser.add_argument('--test',   type=float, dest='test_size', default=0.2,   help='test set fraction (default=0.2)')
parser.add_argument('--splits', type=int,   dest='splits',    default=5,     help='number of splits (default=5)')
parser.add_argument('--eta',    type=float, dest='eta',       default=1e-5,  help='eta hyperparameter (default=1e-5)')
parser.add_argument('--sigma',  type=float, dest='sigma',     default=32.0,  help='sigma hyperparameter (default=32.0)')
parser.add_argument('--kernel', type=str,   dest='kernel',    default='L',   help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default L)')
args = parser.parse_args()
print(vars(args))

train_size = [0.125, 0.25, 0.5, 0.75, 1.0]
debug = False

def main():
  X = np.load(args.repr)
  y = np.loadtxt(args.prop)
  n_rep = args.splits
  eta   = args.eta
  sigma = args.sigma
  test_size = args.test_size
  kernel = get_kernel(args.kernel)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
  all_indexes_train = np.arange(X_train.shape[0])
  K_all  = kernel(X_train, X_train, 1.0/sigma)
  Ks_all = kernel(X_test,  X_train, 1.0/sigma)
  K_all[np.diag_indices_from(K_all)] += eta

  if debug:
    np.random.seed(666)

  for size in train_size:
    size_train = int(np.floor(X_train.shape[0]*size))
    maes = []
    for rep in range(n_rep):
      train_idx = np.random.choice(all_indexes_train, size = size_train, replace=False)
      y_kf_train = y_train[train_idx]
      K  = K_all [train_idx][:,train_idx]
      Ks = Ks_all[:,train_idx]
      alpha = np.linalg.solve(K, y_kf_train)
      y_kf_predict = np.dot(Ks, alpha)
      maes.append(np.mean(np.abs(y_test-y_kf_predict)))
    print("%d\t%e\t%e" % (size_train, np.mean(maes), np.std(maes)))

if __name__ == "__main__":
  main()

