# SPAHM

This code supports the paper
> A. Fabrizio, K. R. Briling, and C. Corminboeuf,<br>
> “SPAHM: the Spectrum of Approximated Hamiltonian Matrices representations”<br>
> [`arXiv:2110.13037 [physics.chem-ph]`](https://arxiv.org/abs/2110.13037)<br>

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
TODO
```

**01.** `code/01_get_properties.py` takes the output of the previous script
and computes the target properties <br> (dipole moment, HOMO, and HOMO–LUMO gap).
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
TODO
```

### 1. Compute the SPAHM representations

**10.** `code/10_get_guess_eigen.py` computes a SPAHM representation for a given molecule.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/10_get_guess_eigen.py [-h] --mol FILENAME --guess GUESS [--basis BASIS]
                                  [--charge CHARGE] [--func FUNC] [--dir DIR]

  -h, --help       show this help message and exit
  --mol FILENAME   path to molecular structure in xyz format
  --guess GUESS    initial guess type
  --basis BASIS    AO basis set
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
TODO
```

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
  --split SPLIT           whether to split the core and valence energies or not
  --dir DIR               directory to save the output in (default=current dir)

```
</details>

*Example:*
```
TODO
```

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
  --kernel KERNEL   kernel type (G for Gaussian and L or myL for Laplacian)
```
</details>

*Example:*
```
TODO
```

### 3. Compute the learning curve


**30.** `code/30_regression.py` computes the learning curve.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/30_regression.py [-h] --x REPR --y PROP [--splits SPLITS] [--eta ETA]
                             [--sigma SIGMA] [--kernel KERNEL]

  -h, --help       show this help message and exit
  --x REPR         path to the representations file
  --y PROP         path to the properties file
  --splits SPLITS  number of splits
  --eta ETA        η hyperparameter
  --sigma SIGMA    σ hyperparameter
  --kernel KERNEL  kernel type (G for Gaussian and L or myL for Laplacian)
```
</details>

*Example:*
```
TODO
```


**31 [optional].** `code/31_final_error.py` prints the full-training prediction error for each molecule.
<details><summary>(click to see the command-line options)</summary>

```
usage: code/31_final_error.py [-h] --x REPR --y PROP [--eta ETA] [--sigma SIGMA]
                              [--kernel KERNEL]

  -h, --help       show this help message and exit
  --x REPR         path to the representations file
  --y PROP         path to the properties file
  --eta ETA        η hyperparameter
  --sigma SIGMA    σ hyperparameter
  --kernel KERNEL  kernel type (G for Gaussian and L or myL for Laplacian)
```
</details>

*Example:*
```
TODO
```





