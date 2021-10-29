import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_kernel

# USER DEFINED
train_size = [0.125, 0.25, 0.5, 0.75, 1.0]
#np.random.seed(666)

parser = argparse.ArgumentParser(description='This program computes the learning curve.')
parser.add_argument('--x', type=str, dest='repr', default="X_eigen_hcore.npy",
                            help='Path to the representations file')
parser.add_argument('--y', type=str, dest='prop', default="../QM7/energies.txt",
                            help="Path to the properties file")
parser.add_argument('--splits', type=int, dest='s_splits', default=5,
                            help='Number of splits')
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
n_rep = args.s_splits
eta   = args.eta
sigma = args.sigma
kernel = get_kernel(args.kernel)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
all_indexes_train = np.arange(X_train.shape[0])

# compute kernels
K_all  = kernel(X_train, X_train, 1.0/sigma)
Ks_all = kernel(X_test,  X_train, 1.0/sigma)
K_all[np.diag_indices_from(K_all)] += eta

for size in train_size:
  size_train = int(np.floor(X_train.shape[0]*size))
  maes = []
  for rep in range(n_rep):
    train_idx = np.random.choice(all_indexes_train, size = size_train, replace=False)
    y_kf_train = y_train[train_idx]
    # kernels
    K  = K_all [train_idx][:,train_idx]
    Ks = Ks_all[:,train_idx]
    # regression
    alpha = np.linalg.solve(K, y_kf_train)
    # predict and estimate the error
    y_kf_predict = np.dot(Ks, alpha)
    maes.append(np.mean(np.abs(y_test-y_kf_predict)))
  print("RESULTS: ",size_train,np.mean(maes),np.std(maes))

