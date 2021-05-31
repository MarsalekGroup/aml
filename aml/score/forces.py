""" Functionality to evaluate score for root mean square deviation of forces."""

__all__ = ["run_rmse_test"]

import pickle
from pathlib import Path

import aml
import numpy as np

from ..constants import angstrom, eV
from .io import load_with_cell
from .util import get_unique_atom_types, get_unique_elements, plot_mae_errors, print_errors


def compute_all_predictions(structures, dir_model, elements):
    """Evaluate the prediction of the committee model."""

    directories = list(Path(dir_model).iterdir())
    n = len(directories)
    fn_template = directories[0] / 'input.nn'

    # Construct MLP object
    n2p2 = aml.N2P2.from_directories(
        directories=directories,
        elements=elements,
        n=n,
        fn_template=fn_template,
        n_tasks=4,
        n_core_task=1,
        remove_output = True
    )

    # Perform prediction
    n2p2.predict(structures)

    return structures


def compute_force_errors(structures, trj):
    """Compute the force errors."""

    # Get reference and committee forces
    def mean(data):
        return np.mean(data, axis=0)
    f_committee_prediction = structures.reduce_property(f_reduce=mean, name_prop='forces', label_prop='predict*')
    f_ref = np.array([s.properties['reference'].forces for s in structures])

    f_rmse_all = {}
    f_ref_all = {}
    f_test_all = {}

    # Determine the unique atom types
    top = trj.topology
    atom_types = get_unique_atom_types(top)

    # units for forces
    u_f = 1000 * angstrom / eV

    for t1 in atom_types:

        # select indices of the atom type
        idx_t1 = top.select('name ' + t1)

        # compute force rmse of this atom type
        f_rmse = np.sqrt(np.mean((f_ref[:, idx_t1] - f_committee_prediction[:, idx_t1])**2)) * u_f

        # root mean squared force for scale
        f_rms = np.sqrt(np.mean((f_ref[:,idx_t1])**2)) * u_f

        f_rmse_all[t1] = [f_rmse, f_rms, f_rmse / f_rms]
        f_ref_all[t1] = f_ref[:, idx_t1] * u_f
        f_test_all[t1] = f_committee_prediction[:, idx_t1] * u_f

    return f_ref_all, f_test_all, f_rmse_all


def run_rmse_test(model_dir, fn_topo, structures, fn_out='force-res.pkl'):
    """Perform the force scoring and save results."""

    # Determine elements and cell from reference topology
    trj_ref = load_with_cell(fn_topo, top=fn_topo)
    elements = get_unique_elements(trj_ref.topology)

    # Run prediction
    structures = compute_all_predictions(structures, model_dir, elements)

    # Compute the errors
    ref_force, test_force, force_errors = compute_force_errors(structures, trj_ref)

    # Plot the errors
    plot_mae_errors(force_errors, fn_out='forces-all.pdf')

    # Print the errors
    print_errors(force_errors)

    # Save results
    results = {
        'ref_force': ref_force,
        'test_force': test_force,
        'force_errors': force_errors
    }
    with open(fn_out, 'wb') as f_out:
        pickle.dump(results, f_out)
