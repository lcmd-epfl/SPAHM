import numpy as np
import scipy 
from pyscf import gto, scf

mol = gto.M(atom='''
O    0.0000000    0.0000000   -0.1653507
H    0.7493682    0.0000000    0.4424329
H   -0.7493682    0.0000000    0.4424329''', basis={'O':'stuttgart_dz', 'H':'minao'}, ecp = {'O':'stuttgart_dz'})

def get_hcore(mol):
    '''Core Hamiltonian'''

    h = mol.intor_symmetric('int1e_kin')
    h+= mol.intor_symmetric('int1e_nuc')

    if len(mol._ecpbas) > 0:
        h += mol.intor_symmetric('ECPscalar')
    return h

def solveF(mol, fock):
    s1e = mol.intor_symmetric('int1e_ovlp')
    return scipy.linalg.eigh(fock, s1e)

fock = get_hcore(mol)

e, c = solveF(mol, fock)

nocc = mol.nelectron // 2
print(e[:nocc])
