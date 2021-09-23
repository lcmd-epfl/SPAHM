import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


lc = np.loadtxt("LC.dat")
lc_vec = np.loadtxt("../eigenvectors/LC.dat")
lc_hamil = np.loadtxt("../H_matrix/LC.dat")
lc_CM = np.loadtxt("../../CM/eigen/LC.dat")

plt.figure(figsize=(10,10),dpi=300)

plt.errorbar(lc[:,0],lc[:,1],yerr=lc[:,2],label='HCORE Occ. Eigenvalues',fmt='.-')
plt.errorbar(lc_vec[:,0],lc_vec[:,1],yerr=lc[:,2],label='HCORE Spherical Projected Eigenvectors',fmt='.-')
plt.errorbar(lc_hamil[:,0],lc_hamil[:,1],yerr=lc[:,2],label='HCORE Spherical Hamiltonian',fmt='.-')
plt.errorbar(lc_CM[:,0],lc_CM[:,1],yerr=lc_CM[:,2],label='Coul. Matrix Eigenvalues',fmt='.-')

plt.legend(loc='upper right',fontsize=10)
#plt.yscale('log')
plt.xscale('log')
plt.yticks([40,30,20], [40,30,20])
plt.xticks([500,1000,10000], [500,1000,10000])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel(r'Test set error [kcal/mol]',fontsize=18)
plt.xlabel(r'Training set size',fontsize=18)
plt.tight_layout()
plt.show()
#plt.savefig("LC_MAE.pdf")

