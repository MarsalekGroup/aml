""" Functionality to evaluate score for Vibrational Density of States"""

__all__ = ["run_vdos_test"]

import pickle

import numpy as np
from scipy import signal

from .util import compute_all_errors, get_unique_atom_types, plot_all_f_and_errors, plot_mae_errors, print_errors


def run_vdos_test(ref_trj, ref_dt, test_trj, test_dt, fn_out='vdos-res.pkl'):
    """Perform the VDOS scoring and save results."""

    # Compute all RDFs
    ref_vdos = compute_all_vdos(ref_trj, ref_dt)
    test_vdos = compute_all_vdos(test_trj, test_dt)

    # Compute the errors
    vdos_errors = compute_all_errors(ref_vdos, test_vdos)

    # Plot the errors
    plot_all_f_and_errors(ref_vdos, test_vdos, vdos_errors, observable='VDOS')
    plot_mae_errors(vdos_errors, fn_out='vdos-all.pdf')

    # Print the errors
    print_errors(vdos_errors)

    # Save results
    results = {
        'ref_vdos': ref_vdos,
        'test_vdos': test_vdos,
        'vdos_errors': vdos_errors
    }
    with open(fn_out, 'wb') as f_out:
        pickle.dump(results, f_out)


def compute_all_vdos(trj, dt=1, Dt=2000.0, Dt_pad=2000.0):
    """Computes VDOS separately for all atom types in a trajectory."""

    top = trj.topology

    vdos_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(top)

    for t1 in atom_types:

        # select indices of the atom type
        idx_t1 = top.select('name ' + t1)

        # calculate velocity autocorrelation functions,
        # averaged over atoms of this species
        cfs = get_acfs(trj.xyz.transpose(1, 0, 2)[idx_t1])

        # calculate the spectrum for this species and store it
        nu, intensity = get_spectra(cfs, dt=dt, Dt=Dt, Dt_pad=Dt_pad)
        vdos_all[t1] = nu, intensity

    return vdos_all


def get_acfs(source):
    """Calculate averaged autocorrelation function for a number of timeseries.

    The `source` yields individual timeseries and each of those is a
    timeseries of vector quantity. This means that if you
    want only a single autocorrelation function, you need to wrap the input
    array in something iterable, like a list.

    Arguments:
        source: iterator over timeseries of dimension N by 3

    Returns:
        CFs, as many ACFs as timeseries yielded by `source`
    """

    acfs = []
    for data in source:
        N = len(data[:, 0])
        norm = N - np.abs(np.arange(1-N, N), dtype=float)
        cfs = [signal.correlate(data[:, d], data[:, d], mode='full', method='auto') / norm
               for d in range(3)]
        acfs.append(np.array(cfs).sum(axis=0))
    acf = np.array(acfs).mean(axis=0)

    return acf


def get_spectra(acfs, dt, Dt, Dt_pad=0.0, f_w=np.hanning):
    """Calculate spectra from autocorrelation functions using FFT.

    Processes multiple spectra at the same time, depending on the data in `acfs`.

    Arguments:
        acfs: CFs numpy array
        dt (float): time step, in femtoseconds
        Dt (float): total (double-sided) width of window, in femtoseconds
        Dt_pad (float): additional (double-sided) width of padding, in femtoseconds
        f_w: function used to generate a (symmetric) window

    Returns:
        nu, intensity
    """

    c = 299792458.0   # m / s

    # check the input
    if Dt <= 0.0:
        raise ValueError('`Dt` must be positive.')
    if Dt_pad < 0.0:
        raise ValueError('`Dt_pad` must not be negative.')
    if dt * len(acfs) < Dt:
        msg = 'The window ({:.0f} fs) must be narrower than the data ({:.0f} fs). Alas, it is not.'
        raise ValueError(msg.format(Dt, dt*len(acfs)))

    nw = int(Dt / dt / 2.0)
    npad = int(Dt_pad / dt / 2.0)

    frequency, intensity = _acfs_to_spectra(acfs, nw, npad=npad, d=1e-15*dt, f_w=f_w)

    # convert frequency from Hz to cm^-1
    nu = frequency / (100.0 * c)

    return nu, intensity


def _acfs_to_spectra(acfs, nw, npad=0, d=1.0, f_w=np.hanning):
    """Calculate spectra from autocorrelation functions using FFT.

    Optionally processes multiple spectra at the same time. In that case,
    `acfs` is a 2D array of shape number of ACFs by length of ACFs. This is a
    low-level function, use `get_spectra` for a more convenient interface.

    Arguments:
        acfs: array, 1D or 2D, symmetric full ACFs
        nw: *one-sided* number of points of ACF window
        npad: *one-sided* number of additional padding zeros
        f_w: function to evaluate the *double-sided* symmetric window

    Returns:
        freq, intensity
    """

    # Make sure we're processing a 2D array.
    ndim = len(acfs.shape)
    if ndim == 1:
        acfs = acfs[np.newaxis, :]
    elif ndim == 2:
        pass
    else:
        raise ValueError('1D or 2D array required.')

    # number of spectra, total length of full ACF
    n, length = acfs.shape

    # window width
    ww = 2 * nw + 1

    assert ww <= length, 'Window cannot be wider than data.'

    # slice ACF data
    data = acfs[:, length//2-nw:length//2+nw+1].copy()
    length_trim = data.shape[1]

    # multiply by the window
    data *= f_w(ww)

    # pad with optional zeros along time axis
    # one extra zero for symmetry - keep the ACF an even function
    data = np.pad(data, ((0, 0), (npad+1, npad)), 'constant', constant_values=0.0)
    assert data.shape == (n, length_trim + 2*npad + 1)

    # window width including zero padding
    wwp = data.shape[1]

    # frequencies, with the provided real-space step
    frequency = np.fft.rfftfreq(wwp, d=d)

    # FFT ACF to spectrum
    # N.B.: For an ACF that is an even function, imaginary part is strictly zero.
    #       This is general, though.
    data_fft = np.fft.rfft(data)
    intensity = np.abs(data_fft)

    # Make result consistent with input in 1D case.
    if ndim == 1:
        intensity = intensity[0, :]

    # Normalize intensities to 1:
    intensity = intensity/intensity.sum()

    return frequency, intensity
