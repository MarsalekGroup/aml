"""Read data specifically produced by i-PI"""

__all__ = ['read_frames_i_pi']

import itertools

from .utilities import Frame, merge_frames, read_frames


def read_frames_i_pi(fn_positions, cell=None, fn_forces=None, fn_energies=None, column_energy=4):
    """Read data specifically produced by i-PI.

    We assume typically used units - angstrom for positions, atomic units for forces and energies.
    i-PI can save data in any units, but we do not attempt to be fully general here. The strides
    if all files are assumed to be the same. For other units or strides, compose the frames by hand
    or write a custom reader function.

    Arguments:
        fn_positions: position trajectory file name, XYZ format
        cell: a constant cell to use in all frames, optional
        fn_forces: forces file name, XYZ format, optional
        fn_energies: energies file name, n2p2 energy format, optional

    Returns:
        a `Frame` object
    """

    # positions from XYZ, we expect units of angstrom for positions from ipi
    frames_pos = read_frames(fn_positions, fformat='xyz')
    frames = [frames_pos]

    # add a constant cell if provided
    if cell is not None:
        frames.append(itertools.repeat(Frame(cell=cell)))

    # add forces from XYZ if filename was provided
    # we expect atomic units for forces from i-PI
    if fn_forces is not None:
        frames.append(read_frames(fn_forces, fformat='xyz', name_data='forces', unit=1.0))

    # add energies from file if filename was provided
    # we expect atomic units for energies from i-PI
    if fn_energies is not None:
        frames.append(read_frames(fn_energies, fformat='N2P2_E', column=column_energy))

    # iterate over merged frames
    yield from merge_frames(*frames)
