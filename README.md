AML
===

Short Description
-----------------

This is a Python package to automatically build the reference set
for the training of _Neural Network Potentials_ (NNPs),
and eventually other machine-learned potentials,
in an automated, data-driven fashion.
For that purpose, a large set of reference configurations
sampled in a physically meaningful way (typically with molecular dynamics)
is filtered and the most important points for the representation of the
_Potential Energy Surface_ (PES) are identified.
This is done by using a set of NNPs, called a committee, for
error estimates of individual configurations.
By iteratively adding the points with the largest
error in the energy/force prediction, the reference
set is progressively extended and optimized.

Keywords:
* Active learning
* Query by committee
* Ensemble averaging
* Committee machines
* Neural Network Potentials

More information can be found in the following references:

* "Machine learning potentials for complex aqueous systems made simple", C.Schran, F.L.Thiemann, P.Rowe, E.A.Müller, O.Marsalek, A.Michaelides _(submitted)_
* "Committee neural network potentials control generalization errors and enable active learning", C.Schran, K.Brezina, O.Marsalek, _J. Chem. Phys._, **153**, 104105 (2020), [10.1063/5.0016004](https://doi.org/10.1063/5.0016004)


Installation
------------

For now, just clone the repository and source the `env.sh` file.

Dependencies:
* NumPy
* SciPy
* Matplotlib
