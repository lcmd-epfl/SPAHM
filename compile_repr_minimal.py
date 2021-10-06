import numpy as np
import os

# USER DEFINED INPUTS
geom_directory = '../../QM7/geometries/'
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

def count_core_val(path):
  atoms = list(np.loadtxt(path, skiprows=2, usecols=[0], dtype=str))
  NperQ = {'H': 2, 'C':10, 'N':10, 'O':10, 'S':18}
  N = 0
  for q in NperQ.keys():
    N += NperQ[q] * atoms.count(q)
  return N//2

N = np.array([count_core_val(geom_directory+i) for i in mol_filenames])

X = np.zeros((len(mol_filenames), max(N)))
for i,x in enumerate(X0):
  X[i,:N[i]] = x[:N[i]]
  print(X[i,:])

np.save('X_eigen_full', X)

