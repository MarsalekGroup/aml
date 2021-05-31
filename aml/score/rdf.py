"""Functionality to evaluate score for radial distribution function."""

__all__ = ["run_rdf_test"]

import pickle

import mdtraj as mdt

from .util import get_unique_atom_types, compute_all_errors, print_errors, plot_all_f_and_errors, plot_mae_errors


def compute_all_rdfs(trj, n_bins=150, **kwargs):
    """Computes the RDFs between all pairs of atom types in a trajectory."""

    top = trj.topology

    rdfs_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(trj.topology)

    for i1, t1 in enumerate(atom_types):

        # select indices of the first atom type
        idx_t1 = top.select('name ' + t1)

        # unique atom type pairs only
        for i2 in range(i1, len(atom_types)):

            t2 = atom_types[i2]

            # select indices of the second atom type
            idx_t2 = top.select('name ' + t2)

            # prepare all pairs of indices
            pairs = trj.topology.select_pairs(idx_t1, idx_t2)

            # single atom with itself -> no RDF
            if len(pairs) == 0:
                continue

            # OM: not sure this should be done here
            min_dimension = trj[0].unitcell_lengths.min() / 2

            r, g_r = mdt.compute_rdf(
                trj, pairs,
                (0, min_dimension), n_bins=n_bins, **kwargs
            )

            rdfs_all[t1 + '-' + t2] = r, g_r

    return rdfs_all


def run_rdf_test(ref_trj, test_trj, fn_out='rdf-res.pkl'):
    """Perform the RDF scoring and save results."""

    # Compute all RDFs
    ref_rdf = compute_all_rdfs(ref_trj)
    test_rdf = compute_all_rdfs(test_trj)

    # Compute the errors
    rdf_errors = compute_all_errors(ref_rdf, test_rdf)

    # Plot the errors
    plot_all_f_and_errors(ref_rdf, test_rdf, rdf_errors, observable='RDF')
    plot_mae_errors(rdf_errors, fn_out='rdf-all.pdf')

    # Print the errors
    print_errors(rdf_errors)

    # Save results
    results = {
        'ref_rdf': ref_rdf,
        'test_rdf': test_rdf,
        'rdf_errors': rdf_errors
    }
    with open(fn_out, 'wb') as f_out:
        pickle.dump(results, f_out)
