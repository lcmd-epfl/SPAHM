#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_kernel,unix_time_decorator

parser = argparse.ArgumentParser(description='This program computes kernel.')
parser.add_argument('--x',      type=str,   dest='repr',      required=True, help='path to the representations file')
parser.add_argument('--sigma',  type=float, dest='sigma',     default=32.0,  help='sigma hyperparameter (default=32.0)')
parser.add_argument('--kernel', type=str,   dest='kernel',    default='L',   help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default L)')
parser.add_argument('--dir',    type=str,   dest='dir',       default='./',   help='directory to save the output in (default=current dir)')
args = parser.parse_args()
print(vars(args))

@unix_time_decorator
def main():
  X = np.load(args.repr)
  sigma = args.sigma
  kernel = get_kernel(args.kernel)
  K = kernel(X, X, 1.0/args.sigma)
  np.save(args.dir+'/K_'+args.kernel, K)

if __name__ == "__main__":
  main()

