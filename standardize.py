import numpy as np

X_core = np.load('X_eigen_core.npy')
X_val  = np.load('X_eigen_val.npy')

bounds = [ -1e6, -52.85, -15.85, -11.50, -8.45, -6.45, 0.0 ]

def standardize(x):
  idx = np.where(x!=0.0)
  if(len(idx[0])>1):
    y = x[idx]
    y = ( y-y.mean() ) / y.std()
    x[idx] = y
  else:
    x[idx] = 1.0

def standardize_column(X0):
  X = np.copy(X0).T
  for x in X:
    standardize(x)
  return X.T

################################################################################

#######
# core
#######

classes = [ [] for _ in range(len(bounds)-1) ]
for x in X_core:
  for i in range(len(classes)):
    classes[i].append (np.where( (x>=bounds[i] ) & (x<bounds[i+1]) )[0])

X_padded = []
for c in classes:
  l = max([len(i) for i in c])
  X_padded.append( np.zeros((len(X_core),l)) )

for j in range(len(X_padded)):
  for i,c in enumerate(classes[j]):
    X_padded[j][i,:len(c)] = X_core[i,c]

# no standardization
X_core_aligned = np.hstack(X_padded)

# standardization within group
for x in X_padded:
  standardize(x)
X_core_st_gr = np.hstack(X_padded)

# standardization within column
X_core_st_col = standardize_column(X_core_aligned)

##########
# valence
##########

# standardization within group
X_val_st_gr = np.copy(X_val)
standardize(X_val_st_gr)

# standardization within column
X_val_st_col = standardize_column(X_val)

#######
# save
#######

np.save('X_core_aligned', X_core_aligned)
np.save('X_core_st_gr',   X_core_st_gr)
np.save('X_core_st_col',  X_core_st_col)

np.save('X_val_st_gr',   X_val_st_gr)
np.save('X_val_st_col',  X_val_st_col)

i=0
for core in [X_core, X_core_aligned, X_core_st_gr, X_core_st_col]:
  for val in [X_val, X_val_st_gr, X_val_st_col]:
    np.save('X_comb_'+str(i), np.hstack((core, val)))
    i+=1

i=0
for core in ['X_core', 'X_core_aligned', 'X_core_st_gr', 'X_core_st_col']:
  for val in ['X_val', 'X_val_st_gr', 'X_val_st_col']:
    print('X_comb_'+str(i), core, val)
    i+=1
