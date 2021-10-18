import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='This program computes the learning curve.')
parser.add_argument('--x', type=str, dest='repr', default="X_eigen_hcore.npy",
                            help='Path to the representations file')
parser.add_argument('--y', type=str, dest='prop', default="../QM7/energies.txt",
                            help="Path to the properties file")
parser.add_argument('--eta', type=float, dest='eta', default=10E-06,
                            help='Eta hyperparameter')
parser.add_argument('--sigma', type=float, dest='sigma', default=31.622776601683793,
                            help='Sigma hyperparameter')
parser.add_argument('--kernel', type=str, dest='kernel', default='L',
                            help='Kernel type (L or G)')
args = parser.parse_args()
print(vars(args))

X = np.load(args.repr)
y = np.loadtxt(args.prop)
eta   = args.eta
sigma = args.sigma
if args.kernel=='G':
  from sklearn.metrics.pairwise import rbf_kernel as kernel
else:
  from sklearn.metrics.pairwise import laplacian_kernel as kernel

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# compute kernels
K_all  = kernel(X_train, X_train, 1.0/sigma)
Ks_all = kernel(X_test,  X_train, 1.0/sigma)
K_all[np.diag_indices_from(K_all)] += eta
alpha = np.linalg.solve(K_all, y_train)
y_kf_predict = np.dot(Ks_all, alpha)

maes = np.abs(y_test-y_kf_predict)
np.savetxt(sys.stdout, maes)

