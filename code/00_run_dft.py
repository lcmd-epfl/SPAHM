#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pyscf import dft
from utils import readmol

parser = argparse.ArgumentParser(description='This program runs a DFT computation for a given molecular system.')
parser.add_argument('--mol',    type=str, dest='filename', required=True,  help='path to molecular structure in xyz format')
parser.add_argument('--basis',  type=str, dest='basis'  ,  required=True,  help='AO basis set')
parser.add_argument('--charge', type=int, dest='charge',   default=0,      help='total charge of the system (default=0)')
parser.add_argument('--func',   type=str, dest='func',     default='pbe0', help='DFT functional (default=PBE0)')
parser.add_argument('--dir',    type=str, dest='dir',      default='./',   help='directory to save the output in (default=current dir)')
args = parser.parse_args()

def main():

  xyz  = args.filename
  name = os.path.basename(xyz).split('.')[0]
  mol  = readmol(xyz, args.basis, charge=args.charge)

  mf = dft.RKS(mol)
  mf.xc = args.func
  mf.verbose = 0
  mf.run()

  f = mf.get_fock()
  d = mf.make_rdm1()
  e = mf.mo_energy
  E = mf.e_tot

  nocc = mol.nelectron // 2
  np.save(args.dir+'/'+name+'_eigens_'+args.func+'_'+args.basis, e[:nocc])
  np.save(args.dir+'/'+name+'_Fock_'+args.func+'_'+args.basis, f)
  np.save(args.dir+'/'+name+'_dm_'+args.func+'_'+args.basis, d)
  print(name, args.func, args.basis, E)

if __name__ == "__main__":
  main()

