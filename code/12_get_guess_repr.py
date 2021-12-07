#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
from utils import readmol,compile_repr
from guesses import *

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a set of molecules.')
parser.add_argument('--geom',   type=str,  dest='geom_directory', required=True,   help='directory with xyz files')
parser.add_argument('--dir',    type=str,  dest='dir',            default='./',    help='directory to save the output in (default=current dir)')
parser.add_argument('--guess',  type=str, dest='guess',           required=True,   help='initial guess type')
parser.add_argument('--basis',  type=str, dest='basis',           default='minao', help='AO basis set (default=MINAO)')
parser.add_argument('--charge', type=int, dest='charge',          default=0,       help='total charge of the system (default=0)')
parser.add_argument('--spin',   type=int, dest='spin',            default=None,    help='number of unpaired electrons (default=None) (use 0 to treat a closed-shell system in a UHF manner)')
parser.add_argument('--func',   type=str, dest='func',            default='hf',    help='DFT functional for the SAD guess (default=HF)')
args = parser.parse_args()

def main():

  guess = get_guess(args.guess)

  geom_directory = args.geom_directory+'/'
  mol_filenames  = sorted(os.listdir(geom_directory))
  mols = []
  for f in mol_filenames:
    print(f)
    mol = readmol(geom_directory+f, args.basis)
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

