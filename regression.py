import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import laplacian_kernel
from qml.math import cho_solve
from scipy.stats import uniform

#### USER DEFINED
train_size = [0.125, 0.25, 0.5, 0.75, 1.0]
n_rep = 5

eta =  10E-06
sigma = 31.622776601683793

# Load features and target
X = np.load("X_eigen_hcore.npy")
y = np.loadtxt("../../../QM7/energies.txt")

# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
all_indexes_train = np.arange(X_train.shape[0])

for size in train_size:
    size_train = int(np.floor(X_train.shape[0]*size))

    maes = []
    for rep in range(n_rep):
        train_idx = np.random.choice(all_indexes_train, size = size_train, replace=False)
        X_kf_train = np.zeros((size_train,X_train.shape[1]))
        y_kf_train = np.zeros(size_train)

        i = 0
        for idx in train_idx:
            X_kf_train[i,:] = X_train[idx,:]
            y_kf_train[i] = y_train[idx]
            i+=1

        # Compute kernels  
        K = laplacian_kernel(X_kf_train, X_kf_train, 1/sigma)
        Ks = laplacian_kernel(X_test, X_kf_train, 1/sigma)

        # Do Regression
        K[np.diag_indices_from(K)] += eta

        alpha = cho_solve(K, y_kf_train)

        # Predict and estimate the error
        y_kf_predict = np.dot(Ks, alpha)

        maes.append(np.mean(np.abs(y_test-y_kf_predict)))

    print("RESULTS: ",size_train,np.mean(maes),np.std(maes))

