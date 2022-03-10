import copy
import numpy
import pyscf.data, pyscf.df

""" Taken from https://github.com/briling/aepm and modified """

class LB2020guess:

  acfile_default  = './parameters_HF.dat'

  def __init__(self, fname=None, parameters='HF'):
    self.get_basis(fname, parameters)

  def renormalize(self, a):
    # 1/norm1 = \int \exp(-a*r^2) d^3 r       => norm1 = (a/pi)^(3/2)
    # 1/norm2^2 = \int (\exp(-a*r^2))^2 d^3 r => norm2 = (2.0*a/pi)^(3/4)
    # coefficient = norm1 / norm2 = (0.5*a/pi)^(3/4)
    x = numpy.sqrt(numpy.sqrt(0.5*a/numpy.pi))
    return x*x*x

  def read_ac(self, fname):
    if fname==None:
      fname = self.acfile_default
    with open(fname) as f:
      lines = f.readlines()
    basis = {'H': []}
    il=0
    while il<len(lines):
      q,ng = map(int,lines[il].split())
      il+=1
      qbasis = []
      for ig in range(ng):
        a,c = map(float,lines[il].split())
        qbasis.append([0,[a, c*self.renormalize(a)]])
        il+=1
      basis[pyscf.data.elements.ELEMENTS[q]] = qbasis
    return basis

  def add_caps(self, basis):
    caps_array = numpy.zeros(103)
    caps_array  [  1 :   2 +1] = 1.0 /  3.0,
    caps_array  [  3 :   4 +1] = 1.0 / 16.0,
    caps_array  [  5 :  10 +1] = 1.0 /  3.0,
    caps_array  [ 11 :  12 +1] = 1.0 / 32.0,
    caps_array  [ 13 :  18 +1] = 1.0 /  8.0,
    caps_array  [ 19 :  20 +1] = 1.0 / 32.0,
    caps_array  [ 21 :  30 +1] = 1.0 /  6.0,
    caps_array  [ 31 :  36 +1] = 1.0 / 12.0,
    caps_array  [ 37 :  38 +1] = 1.0 / 32.0,
    caps_array  [ 39 :  48 +1] = 1.0 /  8.0,
    caps_array  [ 49 :  54 +1] = 1.0 / 12.0,
    caps_array  [ 55 :  70 +1] = 1.0 / 32.0,
    caps_array  [ 71 :  86 +1] = 1.0 / 12.0,
    caps_array  [ 87 : 102 +1] = 1.0 / 32.0
    for q in range(1,103):
      a = caps_array[q]
      qname = pyscf.data.elements.ELEMENTS[q]
      if qname in basis.keys():
        basis[qname].append( [0, [a, self.renormalize(a) ]] )
    return basis

  def get_basis(self, fname, parameters):
    if not parameters:
      acbasis = self.read_ac(fname)
      self.add_caps(acbasis)
      self.acbasis = acbasis
    elif parameters=='HF':
      acbasis = {'H': [],
'He': [[0, [1.8865345899608519, 0.4056146926108746]]],
'Li': [[0, [1.9854870701524918, 0.842937532901041]]],
'Be': [[0, [4.744586184977778, 1.3574437702689057]], [0, [0.2792470137084066, 0.12818229520909]]],
'B': [[0, [6.0338581393756145, 2.094637525409216]], [0, [0.2296652845463048, 0.1538820056563987]]],
'C': [[0, [8.36842382629919, 2.912335066987576]], [0, [0.3175823851018592, 0.2825906903498745]]],
'N': [[0, [10.93399949627562, 3.848864491590766]], [0, [0.43457823405570917, 0.4666119106370673]]],
'O': [[0, [13.822779569568999, 4.823227937581987]], [0, [0.6163807631542392, 0.7589805047258943]]],
'F': [[0, [16.696221288447184, 5.913626376015676]], [0, [0.8069674335184295, 1.1067332169360984]]],
'Ne': [[0, [19.44766524633368, 7.113317051280908]], [0, [1.0081157441421305, 1.508827408605945]]],
'Na': [[0, [22.043514485429395, 8.505970515543133]], [0, [1.0688208368282481, 1.7698621680543754]]],
'Mg': [[0, [35.68089579776235, 9.370720219473146]], [0, [2.9023990296953044, 2.762905547578002]], [0, [0.391911845854857, 0.4395524604043637]]],
'Al': [[0, [34.328377368288, 11.458430900825965]], [0, [1.895391976451897, 2.8294008855907764]], [0, [0.12243916188522636, 0.09609543557391674]]],
'Si': [[0, [40.1763529442365, 13.078588758018803]], [0, [2.239495255980109, 3.340046848460941]], [0, [0.13204220229571037, 0.13836780632028686]]],
'P': [[0, [46.66493733746877, 14.784964733718208]], [0, [2.6279568276824814, 3.8907484994126014]], [0, [0.1594036030260791, 0.2054997181258127]]],
'S': [[0, [54.215297785332154, 16.531442577888246]], [0, [3.167647315145373, 4.489750969396064]], [0, [0.22671769463490918, 0.3487263502408896]]],
'Cl': [[0, [62.03053259370884, 18.377558847158248]], [0, [3.700397336007754, 5.10183267645642]], [0, [0.28974576291563425, 0.5086598918385119]]],
'Ar': [[0, [70.09781762916084, 20.327210911919742]], [0, [4.219331463603571, 5.7261201541458835]], [0, [0.35198503878294074, 0.6852541804555327]]],
'Ni': [[0, [203.1786382537903, 41.69876688913565]], [0, [14.904507246734576, 14.254460551474274]], [0, [2.50518240559692, 4.554224625007843]], [0, [0.4821723138309187, 1.0776113454467862]]],
'Br': [[0, [300.01299028853975, 61.61553868303424]], [0, [17.856104656403406, 23.74892022762549]], [0, [2.1685919518796943, 6.686153971304707]], [0, [0.21670708306814743, 0.39290870861189514]]]}
      self.add_caps(acbasis)
      self.acbasis = acbasis
    elif parameters=='HFS':
      self.acbasis = \
{ 'H': [[0, [0.0815877135278, 0.03846658840144482]]],
'He': [[0, [0.808048051263, 0.42950970838920094]]],
'Li': [[0, [2.60255347642, 0.9236581585938292]], [0, [0.0280604557276, 0.02092188631196157]]],
'Be': [[0, [4.59692793038, 1.5671644720955082]], [0, [0.0804833286681, 0.07687177344753668]]],
'B': [[0, [6.83323506001, 2.343454972959998]], [0, [0.128413097632, 0.15132206888434654]]],
'C': [[0, [9.78271998209, 3.2338673789342076]], [0, [0.220436310973, 0.29830455285829904]]],
'N': [[0, [13.0102305297, 4.234735126785875]], [0, [0.338162891505, 0.5080513541327736]]],
'O': [[0, [16.0693906158, 5.282243530372744]], [0, [0.486361793604, 0.7907181038567846]]],
'F': [[0, [19.101114431, 6.414979114451199]], [0, [0.654163546258, 1.1388759924473113]]],
'Ne': [[0, [21.8775289055, 7.6030507281160205]], [0, [0.840940013903, 1.552754282665946]]],
'Na': [[0, [39.3300572224, 8.771875563897146]], [0, [1.92102415925, 2.714794977091659]], [0, [0.07805961683, 0.0811487825176091]]],
'Mg': [[0, [44.5119316877, 10.28939716988604]], [0, [2.2032684956, 3.187374601857695]], [0, [0.0887611981764, 0.1080039414860704]]],
'Al': [[0, [48.8730920117, 11.90986953428574]], [0, [2.40497322587, 3.637221913420624]], [0, [0.0734291195179, 0.10551672569775308]]],
'Si': [[0, [57.6233652793, 13.49494028809413]], [0, [2.94976481323, 4.320831140260877]], [0, [0.105177215317, 0.17781870789022175]]],
'P': [[0, [66.8662881023, 15.16878399053013]], [0, [3.53343161485, 5.031541131577296]], [0, [0.143855885176, 0.2753173411346297]]],
'S': [[0, [77.3837213998, 16.887417337354126]], [0, [4.24286242552, 5.806114862351204]], [0, [0.19956804901, 0.42411499313349615]]],
'Cl': [[0, [87.9791594478, 18.69938754757193]], [0, [4.96724533871, 6.585294351730177]], [0, [0.25861139087, 0.5996439927014025]]],
'Ar': [[0, [98.6384890866, 20.607909307338506]], [0, [5.70477691943, 7.365867515604668]], [0, [0.322389303278, 0.8050942514147619]]],

  def HLB20(self, mol):
    acbasis = copy.deepcopy(self.acbasis)
    factor = 1.0-mol.charge/mol.natm
    for q in acbasis.keys():
      acbasis[q][-1][1][1] *= factor

    auxmol = pyscf.df.make_auxmol(mol, acbasis)
    pmol  = mol + auxmol
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    iao = 0
    for iat in range(auxmol.natm):
      q = auxmol._atom[iat][0]
      for prim in auxmol._basis[q]:
        eri3c[:,:,iao] *= prim[1][1]
        iao+=1
    return numpy.einsum('pqi->pq', eri3c)

  def Heff(self, mol):
    self.mol = mol
    self.Hcore = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
    self.H    = self.Hcore + self.HLB20(mol)
    return self.H

