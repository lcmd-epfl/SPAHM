import argparse
from pyscf import dft,scf
import numpy as np
from utils import *
import scipy

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program computes properties from a density matrix.')

parser.add_argument('--mol', type=str, dest='filename',
                            help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--basis', type=str, required=True, dest='basis',
                            help="The name of the basis set used for the DM computation.")
parser.add_argument('--charge', type=int, nargs='?', dest='charge', default=0,
                            help='(optional) Total charge of the system (default = 0)')
parser.add_argument('--func', type=str, dest='func', default='pbe0',
                            help='DFT functional (default = pbe0)')

args = parser.parse_args()

########################## Main ##########################


def main():

    xyz_filename = args.filename
    filename = xyz_filename.split('/')[-1].split('.')[0]
    mol = readmol(xyz_filename, args.basis, charge = args.charge)

    path = args.func+'_'+args.basis+'/'
    f = np.load(path+filename+'_Fock_'+args.func+'_'+args.basis+'.npy')
    d = np.load(path+filename+'_dm_'+args.func+'_'+args.basis+'.npy')

    s1e = mol.intor_symmetric('int1e_ovlp')
    e,c = scipy.linalg.eigh(f, s1e)

    nocc = mol.nelectron // 2
    homo = e[nocc-1]
    homolumo = e[nocc]-homo

    dip = scf.hf.dip_moment(mol=mol, dm=d, unit='au', verbose=0)
    dip = np.linalg.norm(dip)

    print(filename, "  % .10e  % .10e  % .10e" % (homo, homolumo, dip))


if __name__ == "__main__":
    main()

