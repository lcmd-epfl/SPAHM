#!/usr/bin/env python3

import os
import argparse
import numpy as np
import qml
from qml.representations import get_slatm_mbtypes, generate_slatm

parser = argparse.ArgumentParser(description='This program generates a standard QML representation for a set of molecules.')
parser.add_argument('--geom',  type=str,  dest='geom_directory', required=True,  help='directory with xyz files')
parser.add_argument('--repr',  type=str,  dest='repr',           required=True,  help='representation (cm or slatm)')
parser.add_argument('--dir',   type=str,  dest='dir',            default='./',   help='directory to save the output in (default=current dir)')
args = parser.parse_args()

def get_CM(mols):
  nmax = max([mol.natoms for mol in mols])
  X = np.zeros((len(mols),nmax))
  for i,mol in enumerate(mols):
    mol.generate_eigenvalue_coulomb_matrix(size=nmax)
    X[i] = mol.representation
  return X

def get_SLATM(mols):
  mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in mols]))
  X = np.array([ generate_slatm(mol.coordinates, mol.nuclear_charges, mbtypes, local=False) for mol in mols ])
  return X

def main():

  reprs = {'cm':get_CM, 'slatm':get_SLATM}
  if args.repr not in reprs.keys():
    print('Unknown representation. Available representations:', list(reprs.keys()));
    exit(1)
  get_repr = reprs[args.repr]

  geom_directory = args.geom_directory+'/'
  mol_filenames  = sorted(os.listdir(geom_directory))
  mols = []
  for i in mol_filenames:
    mols.append(qml.Compound(xyz=geom_directory+i))

  X = get_repr(mols)
  np.save(args.dir+'/X_'+args.repr, X)

if __name__ == "__main__":
  main()

