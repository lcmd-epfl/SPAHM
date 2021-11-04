# SPAHM

This code supports the paper
> A. Fabrizio, K. R. Briling, and C. Corminboeuf,<br>
> “SPAHM: the Spectrum of Approximated Hamiltonian Matrices representations”<br>
> [`arXiv:2110.13037 [physics.chem-ph]`](https://arxiv.org/abs/2110.13037) [`(doi:10.21203/rs.3.rs-1017031/v1)`](https://doi.org/10.21203/rs.3.rs-1017031/v1)<br>

This is a collection of scripts which allow to reproduce the results presented in the paper.

## Requirements
* `python >= 3.6`
* `numpy >= 1.16`
* `scipy >= 1.2`
* `scikit-learn >= 0.20`
* [`pyscf >= 1.6`](https://github.com/pyscf/pyscf)

## Usage

### 0. Compute the target quantum-chemical properties [optional]

**00.** `code/00_run_dft.py` runs a DFT computation
and saves the Fock and density matrices and occupied orbital values.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/00_run_dft.py [-h] --mol FILENAME --basis BASIS [--charge CHARGE]
                          [--func FUNC] [--dir DIR]

  -h, --help       show this help message and exit
  --mol FILENAME   path to molecular structure in xyz format
  --basis BASIS    AO basis set
  --charge CHARGE  total charge of the system (default=0)
  --func FUNC      DFT functional (default=PBE0)
  --dir DIR        directory to save the output in (default=current dir)
```
</details>

*Example:*
```
$ for i in examples/xyz/*.xyz ; do \
    code/00_run_dft.py --mol $i --basis ccpvdz --dir examples/dft/; \
  done
```
computes the Fock and density matrices for the structures in `examples/xyz/` at the PBE0/cc-pVDZ level <br>
and writes them to the `examples/dft/` directory.

***NB: here we use a toy set of only 10 molecules!***

**01.** `code/01_get_properties.py` takes the output of the previous script
and computes the target properties <br> (HOMO, HOMO–LUMO gap, and dipole moment).
<details><summary>(click to see the command-line options)</summary>

```
usage: code/01_get_properties.py [-h] --mol FILENAME --basis BASIS
                                 [--charge CHARGE] [--func FUNC] [--dir DIR]

  -h, --help       show this help message and exit
  --mol FILENAME   path to molecular structure in xyz format
  --basis BASIS    AO basis set
  --charge CHARGE  total charge of the system (default=0)
  --func FUNC      DFT functional (default=PBE0)
  --dir DIR        directory to read the input from (default=current dir)
```
</details>

*Example:*
```
$ for i in examples/xyz/*.xyz ; do \
    code/01_get_properties.py --mol $i --basis ccpvdz --dir examples/dft/; \
  done \
  | tee examples/prop.dat ; awk '{print $4}' examples/prop.dat > examples/dipole.dat
```
writes the dipole moments to `examples/dipole.dat` in the format recognizable by other scripts (plain list).

### 1. Compute the SPAHM representations

**10.** `code/10_get_guess_eigen.py` computes a SPAHM representation for a given molecule.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/10_get_guess_eigen.py [-h] --mol FILENAME --guess GUESS [--basis BASIS]
                                  [--charge CHARGE] [--func FUNC] [--dir DIR]

  -h, --help       show this help message and exit
  --mol FILENAME   path to molecular structure in xyz format
  --guess GUESS    initial guess type
  --basis BASIS    AO basis set (default=MINAO)
  --charge CHARGE  total charge of the system (default=0)
  --func FUNC      DFT functional for the SAD guess (default=HF)
  --dir DIR        directory to save the output in (default=current dir)

Available guesses:  'core'   (the core Hamiltonian),
                    'gwh'    (generalized Wolfsberg–Helmholtz guess),
                    'huckel' (extended Hückel guess),
                    'sad'    (superposition of atomic densities),
                    'sap'    (superposition of atomic potentials),
                    'lb'     (Laikov–Briling SAP-like guess),
                    'lb-hfs' (Laikov–Briling SAP-like guess with HFS-based parameters)
```
</details>

*Example:*
```
$ for i in examples/xyz/*.xyz ; do \
    code/10_get_guess_eigen.py --mol $i --guess lb --dir examples/lb/ ; \
  done
```
computes the SPAM representations based on the LB guess in the MINAO basis for the structures in `examples/xyz/` <br>
and writes them to the `examples/lb/` directory.

**11.** `code/11_compile_repr_core_valence.py` prepares the representations for regression:<br>
pads them with zeros and merges into one file.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/11_compile_repr_core_valence.py [-h] --eig EIG_DIRECTORY
                                            [--geom GEOM_DIRECTORY] [--split SPLIT]
                                            [--dir DIR]

  -h, --help              show this help message and exit
  --eig EIG_DIRECTORY     directory with eigenvalues
  --geom GEOM_DIRECTORY   directory with xyz files
  --split SPLIT           whether to split the core and valence energies or not (default=False)
  --dir DIR               directory to save the output in (default=current dir)

```
</details>

*Example:*
```
$ code/11_compile_repr_core_valence.py --eig examples/lb/ --dir examples/
```
writes the merged representations to `examples/X_lb.npy`.

### 2. Optimize the hyperparameters

**20.** `code/20_hyperparameters.py` optimizes the σ (kernel width) and η (regularization) hyperparameters <br>
using the k-fold cross valigation and grid search.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/20_hyperparameters.py [-h] --x REPR --y PROP [--test TEST_SIZE]
                                  [--splits SPLITS] [--kernel KERNEL]

  -h, --help        show this help message and exit
  --x REPR          path to the representations file
  --y PROP          path to the properties file
  --test TEST_SIZE  test set fraction (default=0.2)
  --splits SPLITS   k in k-fold cross validation (default=5)
  --kernel KERNEL   kernel type (G for Gaussian and L or myL for Laplacian) (default L)
```
</details>

*Example:*
```
$ code/20_hyperparameters.py --x examples/X_lb.npy --y examples/dipole.dat | tail -n 4

5.183039e-01 3.005078e-01 | 1.000000e-05 31.622777
5.182629e-01 3.004739e-01 | 3.162278e-08 31.622777
5.182628e-01 3.004737e-01 | 1.000000e-10 31.622777
5.105925e-01 3.382477e-01 | 1.000000e+00 31.622777
```
performs a grid search for the optimal hyperparameters and prints a sorted list of prediction errors, their standard deviations,
and the corresponding σ and η parameters. Here the optimal parameters are: σ = 32, η = 1e-10.

### 3. Compute the learning curve

**30.** `code/30_regression.py` computes the learning curve.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/30_regression.py [-h] --x REPR --y PROP [--splits SPLITS] [--eta ETA]
                             [--sigma SIGMA] [--kernel KERNEL]

  -h, --help       show this help message and exit
  --x REPR         path to the representations file
  --y PROP         path to the properties file
  --test TEST_SIZE test set fraction (default=0.2)
  --splits SPLITS  number of splits (default=5)
  --eta ETA        η hyperparameter (default=1e-5)
  --sigma SIGMA    σ hyperparameter (default=32.0)
  --kernel KERNEL  kernel type (G for Gaussian and L or myL for Laplacian) (default L)
```
</details>

*Example:*
```
$ code/30_regression.py --x examples/X_lb.npy --y examples/dipole.dat --sigma 32.0 --eta 1e-10

{'repr': 'examples/X_lb.npy', 'prop': 'examples/dipole.dat', 'test_size': 0.2, 'splits': 5, 'eta': 1e-10, 'sigma': 32.0, 'kernel': 'L'}
1       4.508885e-01    1.963332e-01
2       3.743045e-01    2.243524e-01
4       3.685258e-01    2.894607e-01
6       2.480406e-01    1.777623e-01
8       1.939497e-01    6.768116e-15
```
computes the learning curve (σ = 32, η = 1e-10)
and prints the number of training molecules, the prediction error, and its standard deviation <br>
(your learning curve can be different because for the partial training set random shuffling is used).

**31 [optional].** `code/31_final_error.py` prints the full-training prediction error for each molecule.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/31_final_error.py [-h] --x REPR --y PROP [--eta ETA] [--sigma SIGMA]
                              [--kernel KERNEL]

  -h, --help       show this help message and exit
  --x REPR         path to the representations file
  --y PROP         path to the properties file
  --test TEST_SIZE test set fraction (default=0.2)
  --eta ETA        η hyperparameter (default=1e-5)
  --sigma SIGMA    σ hyperparameter (default=32.0)
  --kernel KERNEL  kernel type (G for Gaussian and L or myL for Laplacian) (default L)
```
</details>

*Example:*
```
$ code/31_final_error.py --x examples/X_lb.npy --y examples/dipole.dat --sigma 32.0 --eta 1e-10

{'repr': 'examples/X_lb.npy', 'prop': 'examples/dipole.dat', 'eta': 1e-10, 'test_size': 0.2, 'sigma': 32.0, 'kernel': 'L'}
1.154644e-01
2.724350e-01
```
prints the final prediction error for the test set molecules (just two in this case).

