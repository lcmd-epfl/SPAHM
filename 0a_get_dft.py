import argparse
from pyscf import dft
import numpy as np
from utils import *

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program runs a DFT computation for a given molecular system.')

parser.add_argument('--mol', type=str, dest='filename',
                            help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--basis', type=str, required=True, dest='basis',
                            help="The name of the basis set used for the DM computation.")
parser.add_argument('--charge', type=int, nargs='?', dest='charge', default=0,
                            help='(optional) Total charge of the system (default = 0)')
parser.add_argument('--func', type=str, dest='func', default='b3lyp',
                            help='DFT functional (default = b3lyp)')

args = parser.parse_args()

########################## Main ##########################


def main():

    xyz_filename = args.filename
    filename = xyz_filename.split('/')[-1].split('.')[0]
    mol = readmol(xyz_filename, args.basis, charge = args.charge)

    mf = dft.RKS(mol)
    mf.xc = args.func
    mf.verbose = 0
    mf.run()

    f = mf.get_fock()
    e = mf.mo_energy
    E = mf.e_tot

    nocc = mol.nelectron // 2
    np.save(filename+'_eigens_'+args.func, e[:nocc])
    np.save(filename+'_Fock_'+args.func, f)
    print(filename, args.func, E)


if __name__ == "__main__":
    main()

