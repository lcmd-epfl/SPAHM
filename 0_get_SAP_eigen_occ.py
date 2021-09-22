import argparse
from pyscf import gto
from pyscf.dft import numint
from pyscf import dft
from pyscf.scf import hf
import scipy
import numpy as np

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program computes the core hamiltonian guess for a given molecular system.')

parser.add_argument('--mol', type=str, dest='filename',
                            help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--basis', type=str, required=True, dest='basis', 
                            help="The name of the basis set used for the DM computation.")
parser.add_argument('--charge', type=int, nargs='?', dest='charge', default=0,
                            help='(optional) Total charge of the system (default = 0)')

args = parser.parse_args()

########################## Helper Functions ##########################

def readmol(fin, basis, charge=0):
    """ Read xyz and return pyscf-mol object """ 

    f = open(fin, "r")
    molxyz = '\n'.join(f.read().split('\n')[2:])
    f.close()
    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    mol.charge = charge
    mol.build()

    return mol

def hcore(mol):

    h = mol.intor_symmetric('int1e_kin')
    h+= mol.intor_symmetric('int1e_nuc')

    return h


########################## Main ##########################

def main():
    """ Main """

    # Create pyscf-mol object
    xyz_filename = args.filename
    filename = xyz_filename.split('/')[-1].split('.')[0]
    mol = readmol(xyz_filename, args.basis, charge = args.charge)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    
    dm = dft.rks.init_guess_by_vsap(mf, mol)

    vhf = mf.get_veff(mol, dm)
   
    hc = hcore(mol)
    s1e = mol.intor_symmetric('int1e_ovlp')

    fock = hc + vhf

    # Diagonalize it
    e, v = scipy.linalg.eigh(fock, s1e)

    # 1-assumption repr. occ. eigenvalues.
    nocc = mol.nelectron // 2
    np.save('eigens_'+filename, e[:nocc])


if __name__ == "__main__":
    main()
