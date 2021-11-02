#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
from pyscf import scf
from utils import readmol

parser = argparse.ArgumentParser(description='This program computes target properties using Fock and density matrices.')
parser.add_argument('--mol',    type=str, dest='filename', required=True,  help='path to molecular structure in xyz format')
parser.add_argument('--basis',  type=str, dest='basis'  ,  required=True,  help='AO basis set')
parser.add_argument('--charge', type=int, dest='charge',   default=0,      help='total charge of the system (default=0)')
parser.add_argument('--func',   type=str, dest='func',     default='pbe0', help='DFT functional (default=PBE0)')
parser.add_argument('--dir',    type=str, dest='dir',      default='./',   help='directory to read the input from (default=current dir)')
args = parser.parse_args()

def main():

  xyz  = args.filename
  name = os.path.basename(xyz).split('.')[0]
  mol  = readmol(xyz, args.basis, charge=args.charge)

  f = np.load(args.dir+'/'+name+'_Fock_'+args.func+'_'+args.basis+'.npy')
  d = np.load(args.dir+'/'+name+'_dm_'+args.func+'_'+args.basis+'.npy')

  s1e = mol.intor_symmetric('int1e_ovlp')
  e,c = scipy.linalg.eigh(f, s1e)

  nocc = mol.nelectron // 2
  homo = e[nocc-1]
  homolumo = e[nocc]-homo

  dip = scf.hf.dip_moment(mol=mol, dm=d, unit='au', verbose=0)
  dip = np.linalg.norm(dip)

  print(name, "  % .10e  % .10e  % .10e" % (homo, homolumo, dip))

if __name__ == "__main__":
    main()

