#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
from pyscf import dft,scf
from LB2020guess import LB2020guess
from utils import readmol

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given (closed-shell) molecular system.')
parser.add_argument('--mol',    type=str, dest='filename', required=True,   help='Path to molecular structure in xyz format')
parser.add_argument('--guess',  type=str, dest='guess',    required=True,   help='Initial guess type')
parser.add_argument('--basis',  type=str, dest='basis'  ,  default='minao', help='AO basis set.')
parser.add_argument('--charge', type=int, dest='charge',   default=0,       help='Total charge of the system (default = 0).')
parser.add_argument('--func',   type=str, dest='func',     default='hf',    help='DFT functional for the SAD guess (default = hf).')
parser.add_argument('--dir',    type=str, dest='dir',      default='./',    help='Directory to save the output in (default = current).')
args = parser.parse_args()

def hcore(mol):
  h  = mol.intor_symmetric('int1e_kin')
  h += mol.intor_symmetric('int1e_nuc')
  return h

def GWH(mol):
  h = hcore(mol)
  S = mol.intor_symmetric('int1e_ovlp')
  K = 1.75 # See J. Chem. Phys. 1952, 20, 837
  h_gwh = np.zeros_like(h)
  for i in range(h.shape[0]):
    for j in range(h.shape[1]):
      if i != j:
        h_gwh[i,j] = 0.5 * K * (h[i,i] + h[j,j]) * S[i,j]
      else:
        h_gwh[i,j] = h[i,i]
  return h_gwh

def SAD(mol):
  hc = hcore(mol)
  dm =  scf.hf.init_guess_by_atom(mol)
  mf = dft.RKS(mol)
  mf.xc = args.func
  vhf = mf.get_veff(dm=dm)
  fock = hc + vhf
  return fock

def solveF(mol, fock):
  s1e = mol.intor_symmetric('int1e_ovlp')
  return scipy.linalg.eigh(fock, s1e)

def SAP(mol):
  mf = dft.RKS(mol)
  vsap = mf.get_vsap()
  t = mol.intor_symmetric('int1e_kin')
  fock = t + vsap
  return fock

def LB(mol):
  return LB2020guess(parameters='HF').Heff(mol)

def LB_HFS(mol):
  return LB2020guess(parameters='HFS').Heff(mol)

def main():

  guesses = {'core':hcore, 'sad':SAD, 'sap':SAP, 'gwh':GWH, 'lb':LB, 'huckel':None, 'lb-hfs':LB_HFS}
  if args.guess not in guesses.keys():
    print('Unknown guess. Available guesses:', list(guesses.keys()));
    exit(1)
  guess = guesses[args.guess]

  xyz  = args.filename
  name = os.path.basename(xyz).split('.')[0]
  mol  = readmol(xyz, args.basis, charge=args.charge)

  if args.guess == 'huckel':
    e,v = scf.hf._init_guess_huckel_orbitals(mol)
  else:
    fock = guess(mol)
    e,v = solveF(mol, fock)

  nocc = mol.nelectron // 2
  np.save(args.dir+'/'+args.guess+'_'+args.basis+'_'+name, e[:nocc])
  print(*e[:nocc])

if __name__ == "__main__":
  main()

