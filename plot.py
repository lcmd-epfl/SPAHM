import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('map.txt')

plt.xscale("log")
plt.yscale("log")
plt.scatter(A[:,0],A[:,1], c=A[:,2], vmin=1, vmax=10, cmap='gnuplot')
plt.colorbar()
plt.show()
