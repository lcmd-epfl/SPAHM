import numpy as np
from sklearn.model_selection import train_test_split, KFold
from qml.math import cho_solve
from scipy.stats import uniform
from sklearn.metrics.pairwise import laplacian_kernel

#### USER DEFINED
s_splits = 5 

# Load features and target
X = np.load("X_eigen_hcore.npy")
y = np.loadtxt("../../../QM7/energies.txt")

# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#def manhattan_distances(X, Y):
#    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
#    D = np.abs(D, D)
#    
#    return D.reshape((-1, X.shape[1]))
#
#def laplacian_kernel(Xi, Xj, sigma):
#    K = -1/sigma * manhattan_distances(Xi, Xj)
#    return np.exp(K)


def k_fold_opt(eta, sigma):

   # Do 10-fold cross validation to find the best parameters
   kfold = KFold(n_splits=s_splits, shuffle=False)
   
   all_maes = np.zeros(s_splits)
   i = 0
   for train_idx, test_idx in kfold.split(X_train):
       X_kf_train, X_kf_test = X_train[train_idx], X_train[test_idx]
       y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]
   
       # Compute kernels  
       K = laplacian_kernel(X_kf_train, X_kf_train, 1/sigma)
       Ks = laplacian_kernel(X_kf_test, X_kf_train, 1/sigma)
   
       # Do Regression
       K[np.diag_indices_from(K)] += eta
   
       alpha = cho_solve(K, y_kf_train)
   
       # Predict and estimate the error
       y_kf_predict = np.dot(Ks, alpha)
   
       all_maes[i] = np.mean(np.abs(y_kf_predict - y_kf_test))
       i+=1
       
   print("RESULTS: ",sigma,eta,np.mean(all_maes),np.std(all_maes))
   return np.mean(all_maes)

# Randomized Parameters Search
eta = np.logspace(-10, 0, 5) 
sigma = np.logspace(1,6, 11)

for a in eta:
    for b in sigma:
        k_fold_opt(a, b)


