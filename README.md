AML
===

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/MarsalekGroup/aml)

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

* C. Schran, F. L. Thiemann, P. Rowe, E. A. Müller, O. Marsalek, A. Michaelides,  
  "Machine learning potentials for complex aqueous systems made simple",  
  _PNAS_ **118**, e2110077118 (2021), [10.1073/pnas.2110077118](https://doi.org/10.1073/pnas.2110077118)
* C. Schran, K. Brezina, O. Marsalek,  
  "Committee neural network potentials control generalization errors and enable active learning",  
  _J. Chem. Phys._ **153**, 104105 (2020), [10.1063/5.0016004](https://doi.org/10.1063/5.0016004)

Installation
------------

For now, just clone the repository and source the `env.sh` file.

Dependencies:

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [MDTraj](https://www.mdtraj.org/) (for scoring)
* [cp2k-input-tools](https://github.com/cp2k/cp2k-input-tools) (for CP2K calculations)
