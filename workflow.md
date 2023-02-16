# SPA<sup>H</sup>M

## Full workflow

Let's say that we have a directory with molecular geometries in the `.xyz` format (`examples/xyz/`)
and a file with the target property sorted in alphabetical order (`examples/dipole.dat`).

:warning:
**Use the file names that can be unambiguously sorted (for example, `mol000, mol001, mol002, ... mol998, mol999`).
Other formats (e.g. without leading zeros) may lead to representation-property mismatch.**
:warning:

```
mydir="examples"
Ys="dipole.dat"
Xs="lb cm slatm"

# compute the LB guess
code/12_get_guess_repr.py --geom ${mydir}/xyz/ --dir ${mydir}/ --guess lb
# compute CM and SLATM
code/19_QML-repr.py --geom $mydir/xyz/ --dir $mydir/ --repr cm ;
code/19_QML-repr.py --geom $mydir/xyz/ --dir $mydir/ --repr slatm ;

# hyperparameters
for y in $Ys ; do \
  for repr in $Xs ; do \
    code/20_hyperparameters.py --x $mydir/X_${repr}.npy --y $mydir/${y} > $mydir/hyper_${repr}_${y}.out; \
  done
done
for y in $Ys ; do \
  for repr in $Xs ; do \
    echo -e -n "X_${repr}.npy\t${y}\t"; \
    tail -n 1 $mydir/hyper_${repr}_${y}.out | awk '{print $4"\t"$5}'; \
  done
done > $mydir/hyper.dat

# learning curves
cat $mydir/hyper.dat | while read X Y eta sigma ; do \
  code/30_regression.py --x $mydir/${X} --y $mydir/${Y} --sigma ${sigma} --eta ${eta} > $mydir/lc_${repr}_${y}.out ;
done
```

The learning curves are in `examples/lc_*_*.out`.

