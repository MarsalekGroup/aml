#!/usr/bin/env python

import aml
import aml.score as mlps


# settings - original AIMD trajectory
dt_ref = 2.5
dir_AIMD = '../01-AIMD-zundel-DFTB2/'
fn_trj_ref = dir_AIMD + 'zundel-pos-1.xyz'
fn_frc_ref = dir_AIMD + 'zundel-frc-1.xyz'
fn_vel_ref = dir_AIMD + 'zundel-vel-1.xyz'
fn_topo_ref = dir_AIMD + 'H5O2+-MP2-6311++Gss-10.0A.pdb'

# settings - C-NNP model
dir_model = '../03-create-final-model/final-training/model/'

# settings - C-NNP trajectory
dt_test = 2.5
dir_C_NNP = '../04-C-NNP-MD/'
fn_trj_test = dir_C_NNP + 'zundel-pos-1.xyz'
fn_vel_test = dir_C_NNP + 'zundel-vel-1.xyz'
fn_topo_test = dir_C_NNP + 'H5O2+-MP2-6311++Gss-10.0A.pdb'

# load position trajectory
trj_ref = mlps.load_with_cell(fn_trj_ref, top=fn_topo_ref)
trj_test = mlps.load_with_cell(fn_trj_test, top=fn_topo_test)

# perform RDF scoring
mlps.run_rdf_test(trj_ref, trj_test)

# load velocity trajectory
vel_ref = mlps.load_with_cell(fn_vel_ref, top=fn_topo_ref)
vel_test = mlps.load_with_cell(fn_vel_test, top=fn_topo_test)

# perform VDOS scoring
mlps.run_vdos_test(vel_ref, dt_ref, vel_test, dt_test)

# read AIMD trajectory positions and forces
frames = aml.read_frames_cp2k(fn_positions=fn_trj_ref, fn_forces=fn_frc_ref)
structures_ref = aml.Structures.from_frames(frames, stride=1, probability=1.0)

# perform force RMSE scoring
mlps.run_rmse_test(dir_model, fn_topo_ref, structures_ref)
