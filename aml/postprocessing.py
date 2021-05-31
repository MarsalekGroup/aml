"""Miscellaneous postprocessing tools related to ML potentials.

This might not be the final place for these tools, but it is convenient to have them
available for now. It is not included in the imports of the main package, so import
it with something like:
```
import aml.postprocessing as pp
```
"""

import numpy as np
import matplotlib.pyplot as plt


def load_learning_curve_n2p2(fn_in, n_atoms=1, u_E=1e3*27.21138602, u_l=0.529177210903):
    """Load 'learning_curve.dat' fron n2p2.

    Arguments:
        fn_in: name of data file
        n_atoms: number of atoms per molecule
        u_E: unit of energy, relative to meV
        u_l: unit of length, relative to angstrom
    """

    data = np.loadtxt(fn_in)

    # structure data nicely, change units
    learning_curve = {
        'epoch': data[:, 0],
        'RMSE E': {
            'train': data[:, 1] * u_E * n_atoms,
            'test': data[:, 2] * u_E * n_atoms
        },
        'RMSE F': {
            'train': data[:, 3] * u_E / u_l,
            'test': data[:, 4] * u_E / u_l
        },
        'n_atoms': n_atoms
    }

    return learning_curve


def plot_learning_curve(
        learning_curves, label, label_ref=None,
        xlim=(None, None), ylim_E=(None, None), ylim_F=(None, None),
        do_logscale=False, do_title=None, filename=None):
    """Plot a learning curve.

    More specifically, four curves stored in a dictionary, with "epoch" on the x axis.
    Optionally, a reference curve (test set data) can be shown in gray, and various limits can be set.
    """

    def plot_panel(lc, label, lc_ref, unit, ylabel, ylim):
        if lc_ref is not None:
            epoch = lc_ref['epoch']
            RMSE = lc_ref[label]
            plt.plot(epoch, RMSE['test'], label='test, reference', color='0.75')
        epoch = lc['epoch']
        RMSE = lc[label]
        for t in 'train', 'test':
            idx = np.argmin(RMSE[t])
            label = t + ': {:.2f} {:s}'.format(RMSE[t][idx], unit)
            line, = plt.plot(epoch, RMSE[t], label=label)
            plt.plot(epoch[idx], RMSE[t][idx], 'o', color=line.get_color(), lw=1.0)
        plt.legend()
        plt.ylabel(ylabel)
        if do_logscale:
            plt.loglog()
        plt.xlim(*xlim)
        plt.ylim(*ylim)

    # get the learning curve
    learning_curve = learning_curves[label]

    # get reference learning curve, if any
    if (label_ref is not None) and (label_ref != label):
        lc_ref = learning_curves[label_ref]
    else:
        lc_ref = None

    # determine name of system chunk for energy error
    if learning_curve['n_atoms'] == 1:
        name_chunk = 'atom'
    else:
        name_chunk = 'molecule'

    figure, axes = plt.subplots(2, 1, constrained_layout=True)

    plt.sca(axes[0])
    plot_panel(
        learning_curve, 'RMSE E', lc_ref,
        unit=f'meV/{name_chunk:s}', ylabel=f'Energy RMSE [meV/{name_chunk:s}]', ylim=ylim_E
    )
    plt.tick_params(labelbottom=False)

    plt.sca(axes[1])
    plot_panel(
        learning_curve, 'RMSE F', lc_ref,
        unit='meV/$\mathrm{{\AA}}$', ylabel='Force RMSE [meV/$\mathrm{\AA}$]', ylim=ylim_F
    )

    if do_title is not None:
        title = label
        if label_ref is not None:
            title += f' | reference: {label_ref:s}'
        plt.suptitle(title)
    plt.xlabel('Epoch')

    if filename is not None:
        plt.savefig(filename)
