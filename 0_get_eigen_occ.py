import argparse
from pyscf import gto,dft
from pyscf.dft import numint
from pyscf.scf import hf
import scipy
import numpy as np
from pyscf.scf.hf import _init_guess_huckel_orbitals
from LB2020guess import LB2020guess

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program computes the core hamiltonian guess for a given molecular system.')

parser.add_argument('--mol', type=str, dest='filename',
                            help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--basis', type=str, required=True, dest='basis',
                            help="The name of the basis set used for the DM computation.")
parser.add_argument('--charge', type=int, nargs='?', dest='charge', default=0,
                            help='(optional) Total charge of the system (default = 0)')
parser.add_argument('--guess', type=str, dest='guess', required=True,
                            help='Initial guess type')
parser.add_argument('--func', type=str, dest='func', default='b3lyp',
                            help='DFT functional (default = b3lyp)')

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
                h_gwh[i,j] = 0.5 * K * (h[i,i] + h[j,j]) *S[i,j]
            else:
                h_gwh[i,j] = h[i,i]

    return h_gwh

def SAD(mol):
    hc = hcore(mol)
    dm =  hf.init_guess_by_atom(mol)
    vhf = hf.get_veff(mol, dm)
    fock = hc + vhf
    return fock

def solveF(mol, fock):
    s1e = mol.intor_symmetric('int1e_ovlp')
    return scipy.linalg.eigh(fock, s1e)

def SAP_dm(mol):
    mf = dft.RKS(mol)
    mf.xc = args.func
    dm = dft.rks.init_guess_by_vsap(mf, mol)
    vhf = mf.get_veff(mol, dm)
    hc = hcore(mol)
    fock = hc + vhf
    return fock

def SAP(mol):
    mf = dft.RKS(mol)
    vsap = mf.get_vsap()
    t = mol.intor_symmetric('int1e_kin')
    fock = t + vsap
    return fock

def huckel():
    return None

def LB(mol):
    return LB2020guess().Heff(mol)

########################## Main ##########################


def main():
    """ Main """

    # Create pyscf-mol object
    xyz_filename = args.filename
    filename = xyz_filename.split('/')[-1].split('.')[0]
    mol = readmol(xyz_filename, args.basis, charge = args.charge)

    guesses = {'core':hcore, 'sad':SAD, 'sap':SAP, 'sap-dm':SAP_dm, 'gwh':GWH, 'lb':LB, 'huckel':huckel}

    if args.guess not in guesses.keys():
      print("Unknown guess. Available guesses:", list(guesses.keys()));
      exit(0)
    else:
      guess = guesses[args.guess]

    if args.guess == 'huckel':
        e,v = _init_guess_huckel_orbitals(mol)
    else:
        fock = guess(mol)
        e,v = solveF(mol, fock)

    # 1-assumption repr. occ. eigenvalues.
    nocc = mol.nelectron // 2
    np.save('eigens_'+filename, e[:nocc])
    print(*e[:nocc])


if __name__ == "__main__":
    main()

