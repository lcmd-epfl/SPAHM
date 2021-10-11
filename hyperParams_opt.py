import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# USER DEFINED
eta = np.logspace(-10, 0, 5)
sigma = np.logspace(0,6, 13)

parser = argparse.ArgumentParser(description='This program finds the optimal hyperparameters.')
parser.add_argument('--x', type=str, dest='repr', default="X_eigen_hcore.npy",
                            help='Path to the representations file')
parser.add_argument('--y', type=str, dest='prop', default="../QM7/energies.txt",
                            help="Path to the properties file")
parser.add_argument('--test', type=float, dest='test_size', default=0.2,
                            help='Test set fraction')
parser.add_argument('--splits', type=int, dest='s_splits', default=5,
                            help='Number of splits')
parser.add_argument('--kernel', type=str, dest='kernel', default='L',
                            help='Kernel type (L or G)')
args = parser.parse_args()
print(vars(args))

X = np.load(args.repr)
y = np.loadtxt(args.prop)
test_size = args.test_size
s_splits  = args.s_splits
if args.kernel=='G':
  from sklearn.metrics.pairwise import rbf_kernel as kernel
else:
  from sklearn.metrics.pairwise import laplacian_kernel as kernel

def k_fold_opt(eta, sigma):
   K_all = kernel(X_train, X_train, 1.0/sigma)
   K_all[np.diag_indices_from(K_all)] += eta
   # Do k-fold cross validation to find the best parameters
   kfold = KFold(n_splits=s_splits, shuffle=False)
   all_maes = np.zeros(s_splits)
   i = 0
   for train_idx, test_idx in kfold.split(X_train):
       y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]
       # Take kernels
       K  = K_all[train_idx][:,train_idx]
       Ks = K_all[test_idx ][:,train_idx]
       # Do Regression
       alpha = np.linalg.solve(K, y_kf_train)
       # Predict and estimate the error
       y_kf_predict = np.dot(Ks, alpha)
       all_maes[i] = np.mean(np.abs(y_kf_predict - y_kf_test))
       i+=1
   mean = np.mean(all_maes)
   std  = np.std(all_maes)
   print("RESULTS: ", sigma, eta, mean, std)
   return mean, std, eta, sigma

# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

errors = []
for a in eta:
  for b in sigma:
    errors.append(k_fold_opt(a, b))
errors = np.array(errors)

ind = np.argsort(errors[:,0])[::-1]
print()
print('error        stdev          eta          sigma')
for i in ind:
  print("%e %e | %e %f" % tuple(errors[i]))

