#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from pyscf import scf
from utils import readmol,compile_repr
from guesses import *

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a set of molecules.')
parser.add_argument('--geom',   type=str,  dest='geom_directory', required=True,   help='directory with xyz files')
parser.add_argument('--guess',  type=str, dest='guess',           required=True,   help='initial guess type')
parser.add_argument('--basis',  type=str, dest='basis',           default='minao', help='AO basis set (default=MINAO)')
parser.add_argument('--charge', type=str, dest='charge',          default=None,    help='file with a list of charges')
parser.add_argument('--spin',   type=str, dest='spin',            default=None,    help='file with a list of numbers of unpaired electrons')
parser.add_argument('--func',   type=str, dest='func',            default='hf',    help='DFT functional for the SAD guess (default=HF)')
parser.add_argument('--dir',    type=str,  dest='dir',            default='./',    help='directory to save the output in (default=current dir)')
args = parser.parse_args()

def get_chsp(f, n):
  if f:
    chsp = np.loadtxt(f, dtype=int)
    if(len(chsp)!=n):
      print('Wrong lengh of the file', f, file=sys.stderr);
      exit(1)
  else:
    chsp = np.zeros(n, dtype=int)
  return chsp

def main():

  guess = get_guess(args.guess)

  geom_directory = args.geom_directory+'/'
  mol_filenames  = sorted(os.listdir(geom_directory))
  spin   = get_chsp(args.spin,   len(mol_filenames))
  charge = get_chsp(args.charge, len(mol_filenames))

  mols = []
  for i,f in enumerate(mol_filenames):
    print(f)
    mol = readmol(geom_directory+f, args.basis, charge=charge[i], spin=spin[i])
    mols.append(mol)

  X0 = []
  lens = []
  for mol in mols:
    if args.guess == 'huckel':
      e,v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
      fock = guess(mol)
      e,v = solveF(mol, fock)
    x = get_occ(e, mol.nelec, args.spin)
    X0.append(x)
    lens.append(x.shape)

  X = compile_repr(X0, lens)
  np.save(args.dir+'/X_'+args.guess, X)


if __name__ == "__main__":
  main()

