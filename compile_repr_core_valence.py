import numpy as np
import os

# USER DEFINED INPUTS
geom_directory = '../QM7/geometries/'
eig_directory = './eigens/'

mol_filenames = sorted(os.listdir(geom_directory))
eig_filenames = sorted(os.listdir(eig_directory))

X0   = []
lens = []
for rep in eig_filenames:
  x = np.load(eig_directory+rep)
  X0.append(x)
  lens.append(len(x))
#  print(x)

X = np.zeros((len(X0), max(lens)))

for i,x in enumerate(X0):
  X[i,0:lens[i]] = x

#print(X.shape)
np.save('X_eigen_occ', X)

def count_core_val(path):
  atoms = list(np.loadtxt(path, skiprows=2, usecols=[0], dtype=str))
  NperQ = {'H': (0,1), 'C':(2,4), 'N':(2,5), 'O':(2,6), 'S':(10,6)}
  N = np.array([0,0])
  for q in NperQ.keys():
    N += np.array(NperQ[q]) * atoms.count(q)
  return N//2

N = []
for i in mol_filenames:
  n = count_core_val(geom_directory+i)
  N.append(n)
N = np.array(N)

Xcore = np.zeros((len(X0), max(N[:,0])))
Xval  = np.zeros((len(X0), max(N[:,1])))

for i,x in enumerate(X0):
  ncore,nval = N[i]
  Xcore[i,:ncore] = x[:ncore]
  Xval [i,:nval ] = x[ncore:]

Xcoreval = np.concatenate((Xcore, Xval), axis=1)

np.save('X_eigen_core',    Xcore)
np.save('X_eigen_val',     Xval)
np.save('X_eigen_coreval', Xcoreval)
