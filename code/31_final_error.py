#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_kernel

parser = argparse.ArgumentParser(description='This program computes the full-training error for each molecule.')
parser.add_argument('--x',      type=str,   dest='repr',      required=True, help='path to the representations file')
parser.add_argument('--y',      type=str,   dest='prop',      required=True, help='path to the properties file')
parser.add_argument('--eta',    type=float, dest='eta',       default=1e-5,  help='eta hyperparameter')
parser.add_argument('--sigma',  type=float, dest='sigma',     default=32.0,  help='sigma hyperparameter')
parser.add_argument('--kernel', type=str,   dest='kernel',    default='L',   help='kernel type (G for Gaussian and L or myL for Laplacian)')
args = parser.parse_args()
print(vars(args))

def main():
  X = np.load(args.repr)
  y = np.loadtxt(args.prop)
  eta   = args.eta
  sigma = args.sigma
  kernel = get_kernel(args.kernel)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  K_all  = kernel(X_train, X_train, 1.0/sigma)
  Ks_all = kernel(X_test,  X_train, 1.0/sigma)
  K_all[np.diag_indices_from(K_all)] += eta
  alpha = np.linalg.solve(K_all, y_train)
  y_kf_predict = np.dot(Ks_all, alpha)
  maes = np.abs(y_test-y_kf_predict)
  np.savetxt(sys.stdout, maes)

if __name__ == "__main__":
  main()

