#!/usr/bin/env python3

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program pads the representations with zeros.')
parser.add_argument('--eig',   type=str,  dest='eig_directory',  required=True,  help='directory with eigenvalues')
parser.add_argument('--geom',  type=str,  dest='geom_directory', default=None,   help='directory with xyz files')
parser.add_argument('--split', type=bool, dest='split',          default=False,  help='whether to split the core and valence energies or not (default=False)')
parser.add_argument('--dir',   type=str,  dest='dir',            default='./',   help='directory to save the output in (default=current dir)')
args = parser.parse_args()

def count_core_val(path):
  atoms = list(np.loadtxt(path, skiprows=2, usecols=[0], dtype=str))
  NperQ = {'H': (0,1), 'C':(2,4), 'N':(2,5), 'O':(2,6), 'S' :(10,6),
           '1': (0,1), '6':(2,4), '7':(2,5), '8':(2,6), '16':(10,6)}
  N = np.array([0,0])
  for q in NperQ.keys():
    N += np.array(NperQ[q]) * atoms.count(q)
  return N//2

def main():

  if args.split==True and args.geom_directory==None:
    print('Please specify the geometries directory')
    return

  eig_directory = args.eig_directory+'/'
  eig_filenames = sorted(os.listdir(eig_directory))
  name = os.path.basename(os.path.dirname(eig_directory))

  X0   = []
  lens = []
  for rep in eig_filenames:
    x = np.load(eig_directory+rep)
    X0.append(x)
    lens.append(x.shape)

  X = np.zeros((len(X0), *max(lens)))
  if len(X.shape)==2:
    for i,x in enumerate(X0):
      X[i,0:lens[i][-1]] = x
  else:
    for i,x in enumerate(X0):
      X[i,:,0:lens[i][-1]] = x
  np.save(args.dir+'/X_'+name, X)

  if args.split==False:
    return

  geom_directory = args.geom_directory+'/'
  mol_filenames  = sorted(os.listdir(geom_directory))
  N = np.array([count_core_val(geom_directory+mol) for mol in mol_filenames])

  Xcore = np.zeros((len(X0), max(N[:,0])))
  Xval  = np.zeros((len(X0), max(N[:,1])))
  for i,x in enumerate(X0):
    ncore,nval = N[i]
    Xcore[i,:ncore] = x[:ncore]
    Xval [i,:nval ] = x[ncore:]
  Xcoreval = np.concatenate((Xcore, Xval), axis=1)

  np.save(args.dir+'/X_'+name+'_core',    Xcore)
  np.save(args.dir+'/X_'+name+'_val',     Xval)
  np.save(args.dir+'/X_'+name+'_coreval', Xcoreval)

if __name__ == "__main__":
  main()

