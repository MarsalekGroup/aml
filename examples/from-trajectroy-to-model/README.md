# From Trajectory to Model

This example shows the development of a C-NNP model from a single reference AIMD trajectory. We use the Zundel cation in vacuum described at the DFTB level as a simple demonstration system.


## Prerequisites

* AML itself.
* CP2K version 8.1 - freely available at [CP2K](https://www.cp2k.org). The precompiled binary, for example, is sufficient for this purpose.
* n2p2 - freely available at [n2p2](https://github.com/CompPhysVienna/n2p2).


## Procedure

The process is split into five consecutive steps in the numbered directories:
1. `01-AIMD-zundel-DFTB2`: run AIMD to generate the reference trajectory (use CP2K)
2. `02-QbC`: perform the active learning cycle (query by committee) to generate the training set
3. `03-create-final-model`: train the final C-NNP model using the above training set
4. `04-C-NNP-MD`: apply the final C-NNP model in more extensive MD simulations (use CP2K)
5. `05-validation`: perform automated validation of C-NNP model
