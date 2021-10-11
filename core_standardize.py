import numpy as np

X = np.load('X_eigen_core.npy')

bounds = [ -1e6, -52.85, -15.85, -11.50, -8.45, -6.45, 0.0 ]

classes = []
for i in range(len(bounds)-1):
  classes.append([])

for x in X:
  for i in range(len(classes)):
    classes[i].append (np.where( (x>=bounds[i] ) & (x<bounds[i+1]) )[0])

X_padded = []
for c in classes:
  l = max([len(i) for i in c])
  X_padded.append( np.zeros((len(X),l)) )

for j in range(len(X_padded)):
  for i,c in enumerate(classes[j]):
    X_padded[j][i,:len(c)] = X[i,c]

# v1: no standardization

X_new_v1 = np.hstack(X_padded)

# v2: standardization within group

for x in X_padded:
  idx = np.where(x!=0.0)
  y = x[idx]
  y = ( y-y.mean() ) / y.std()
  x[idx] = y
X_new_v2 = np.hstack(X_padded)

np.save('X_new_v1', X_new_v1)
np.save('X_new_v2', X_new_v2)
