"""Scoring specific Input/Output functions."""

__all__ = [
    "load_with_cell",
    "change_dcd_header"
]

import mdtraj as mdt


def load_with_cell(filename_or_filenames, start=None, stop=None, step=None, **kwargs):
    """Load a trajectory and inject cell dimensions from a topology PDB file if not present.

    All arguments and keyword arguments are passed on to `mdtraj.load`. The `top`
    keyword argument is used load a PDB file and get cell information from it.
    """

    # load the "topology frame" to get cell dimensions
    top = kwargs.get("top")
    if top is not None and isinstance(top, str):
        # load topology frame - just the first one from the file, in case there are more frames
        frame_top = mdt.load_frame(top, 0)
        unitcell_lengths = frame_top.unitcell_lengths
        unitcell_angles = frame_top.unitcell_angles
        if (unitcell_lengths is None) or (unitcell_angles is None):
            raise ValueError("Frame providing topology is missing cell information.")
    else:
        raise ValueError("Provide a PDB with cell dimensions.")

    # load the trajectory itself
    trj = mdt.load(filename_or_filenames, **kwargs)
    trj = trj[start:stop:step]

    # inject the cell information
    len_trj = len(trj)
    trj.unitcell_lengths = unitcell_lengths.repeat(len_trj, axis=0)
    trj.unitcell_angles = unitcell_angles.repeat(len_trj, axis=0)

    return trj


def change_dcd_header(filename, target='CORD', verbose=False):
    """Change the header of a DCD file.

    This is especially useful for DCD files with velocities which are to be read by MDTraj.
    In that case, the header needs to be changed to 'CORD' (the default).
    """

    target = target.encode()

    if len(target) != 4:
        raise ValueError('The new value has to be 4 bytes.')

    if verbose:
        print(f'Target value: {target}')

    with open(filename, 'rb+') as f:
        if verbose:
            f.seek(4)
            value = f.read(4)
            print(f'Current value: {value}')
        f.seek(4)
        f.write(target)

    if verbose:
        print('File changed.')
