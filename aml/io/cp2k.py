"""Functions for CP2K-specific input/output."""

__all__ = ['add_energy_cp2k_comment', 'read_frames_cp2k']

from itertools import repeat

from .utilities import Frame, merge_frames, read_frames


def add_energy_cp2k_comment(frames):
    """Parse CP2K energy and inject it into frames.

    For each frame in `frames`, try to extract a CP2K-formatted potential energy
    from the comment string and inject it back into the frame. Energy from CP2K is
    in Hartree, so no conversion is needed.
    """

    for frame in frames:

        if frame.energy is not None:
            raise ValueError('Energy already present.')

        try:
            for pair in frame.comment.split(','):
                items = pair.split('=')
                if items[0].strip() == 'E':
                    frame.energy = float(items[1])
                    break
        except (IndexError, ValueError):
            raise ValueError('No CP2K energy found in comment line.')

        yield frame


def read_frames_cp2k(fn_positions, cell=None, fn_forces=None, read_energy: bool = True, force_unit=1.0):
    """Read data specifically produced by CP2K.

    Arguments:
        fn_positions: position trajectory file name, XYZ format
        cell: a constant cell to use in all frames, optional
        fn_forces: forces file name, XYZ format, optional
        read_energy: whether to read energies from comments in `fn_positions`

    Returns:
        a `Frame` object
    """

    # positions from XYZ, energies from comment if requested
    # we expect units of angstrom for positions from CP2K
    frames_pos = read_frames(fn_positions, fformat='xyz')
    if read_energy:
        frames_pos = add_energy_cp2k_comment(frames_pos)
    frames = [frames_pos]

    # add a constant cell if provided
    if cell is not None:
        frames.append(repeat(Frame(cell=cell)))

    # add forces from XYZ if filename was provided
    # we expect atomic units for forces from CP2K per default
    if fn_forces is not None:
        frames.append(read_frames(fn_forces, fformat='xyz', name_data='forces', unit=force_unit))

    # iterate over merged frames
    yield from merge_frames(*frames)
