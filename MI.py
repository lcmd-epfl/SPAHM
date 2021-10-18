import numpy as np

X_core = np.load('repr/X_eigen_core.npy')
X_val  = np.load('repr/X_eigen_val.npy')
X_occ  = np.load('repr/X_eigen_occ.npy')

X_core_aligned = np.load('repr/X_core_aligned.npy')
X_core_st_gr   = np.load('repr/X_core_st_gr.npy')
X_core_st_col  = np.load('repr/X_core_st_col.npy')

X_val_st_gr    = np.load('repr/X_val_st_gr.npy')
X_val_st_col   = np.load('repr/X_val_st_col.npy')

d = np.loadtxt('../QM7/dipole.txt')
e = np.loadtxt('../QM7/energies.txt')
g = np.loadtxt('../QM7/gap.txt')
h = np.loadtxt('../QM7/homo.txt')

from sklearn.feature_selection import mutual_info_regression as mi

for x in ['X_core', 'X_val', 'X_occ', 'X_core_aligned', 'X_core_st_gr', 'X_core_st_col', 'X_val_st_gr', 'X_val_st_col']:
  X = eval(x)
  m = []
  m1 = []
  for Y in [e, d, h, g]:
    m.append(mi(X, Y))
    t = []
    for xi in X.T:
      idx = np.where(xi!=0.0)
      t.append(mi(xi[idx].reshape(-1, 1),Y[idx])[0])
    m1.append(t)
  m  = np.array(m).T
  m1 = np.array(m1).T
  np.savetxt('MI_'+x+'.dat', m)
  np.savetxt('MI_renorm_'+x+'.dat', m1)
