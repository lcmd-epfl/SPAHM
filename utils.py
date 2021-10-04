import pyscf

def readmol(fin, basis, charge=0):
    """ Read xyz and return pyscf-mol object """

    f = open(fin, "r")
    molxyz = '\n'.join(f.read().split('\n')[2:])
    f.close()
    mol = pyscf.gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    mol.charge = charge
    mol.build()

    return mol
