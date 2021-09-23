import numpy as np
import os

##### USER DEFINED INPUTS
geom_directory = './eigens/'
size = 33 

# Load database
mol_filenames = sorted(os.listdir(geom_directory))

# Generate and save coulomb matrix representation of the DB
X = []
for rep in mol_filenames:
    tmp = np.load(geom_directory+rep)
    size_pad = size - tmp.shape[0]
    tmp = np.pad(tmp, (0,size_pad), 'constant', constant_values=0)
    print(tmp)
    X.append(tmp)

X = np.array(X)
print(X.shape)
np.save('X_eigen_hcore',X)


