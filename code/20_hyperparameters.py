#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from utils import get_kernel

parser = argparse.ArgumentParser(description='This program finds the optimal hyperparameters.')
parser.add_argument('--x',      type=str,   dest='repr',      required=True, help='path to the representations file')
parser.add_argument('--y',      type=str,   dest='prop',      required=True, help='path to the properties file')
parser.add_argument('--test',   type=float, dest='test_size', default=0.2,   help='test set fraction (default=0.2)')
parser.add_argument('--splits', type=int,   dest='splits',    default=5,     help='k in k-fold cross validation (default=5)')
parser.add_argument('--kernel', type=str,   dest='kernel',    default='L',   help='kernel type (G for Gaussian and L or myL for Laplacian) (default=L)')
args = parser.parse_args()
print(vars(args))

eta   = np.logspace(-10, 0, 5)
sigma = np.logspace(0,6, 13)

X = np.load(args.repr)
y = np.loadtxt(args.prop)
test_size = args.test_size
splits = args.splits
kernel = get_kernel(args.kernel)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

def k_fold_opt(eta, sigma):
  K_all = kernel(X_train, X_train, 1.0/sigma)
  K_all[np.diag_indices_from(K_all)] += eta
  kfold = KFold(n_splits=splits, shuffle=False)
  all_maes = []
  for train_idx, test_idx in kfold.split(X_train):
    y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]
    K  = K_all[train_idx][:,train_idx]
    Ks = K_all[test_idx ][:,train_idx]
    alpha = np.linalg.solve(K, y_kf_train)
    y_kf_predict = np.dot(Ks, alpha)
    all_maes.append(np.mean(np.abs(y_kf_predict-y_kf_test)))
  mean = np.mean(all_maes)
  std  = np.std(all_maes)
  print(sigma, eta, mean, std)
  return mean, std, eta, sigma

errors = []
for e in eta:
  for s in sigma:
    errors.append(k_fold_opt(e, s))
errors = np.array(errors)

ind = np.argsort(errors[:,0])[::-1]
print()
print('error        stdev          eta          sigma')
for i in ind:
  print("%e %e | %e %f" % tuple(errors[i]))

