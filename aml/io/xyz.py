"""Functions to read and write XYZ files."""

__all__ = [
    'write_frame_xyz',
    'read_frame_xyz',
]

import numpy as np

from ..constants import angstrom
from .utilities import Frame, register_io


@register_io('xyz', 'read', 'xyz')
def read_frame_xyz(f_in, name_data='positions', unit=angstrom):
    """Read one frame of XYZ format from an open file.

    Arguments:
        f_in: open file in XYZ format
        name_data: what quantity to take the XYZ data as
        unit: unit to scale data by, multiplicative factor in atomic units

    Returns:
        `Frame` object or `None` if there is no more data
    """

    # read first line to examine it
    line_begin = f_in.readline()

    # no more data in the file
    if not line_begin:
        return None

    # there is some data, frame should begin with natoms
    natoms = int(line_begin)

    # read comment line
    comment = f_in.readline().rstrip()

    names = []
    data = []
    for _ in range(natoms):
        line = f_in.readline()
        if line.strip() == '':
            raise ValueError('Unexpected data in file.')
        items = line.split()
        names.append(items[0])
        data.append([float(item) for item in items[1:4]])
    data = np.array(data) * unit

    # so unless the code fails, this will not trigger.
    if len(names) != natoms:
        raise ValueError('Inconsistent number of atoms in XYZ file.')

    # prepare data
    if name_data == 'positions':
        positions = data
        forces = None
    elif name_data == 'forces':
        positions = None
        forces = data
    else:
        raise ValueError(f'Unsupported `name_data`: {name_data}. Expected "positions" or "forces".')

    return Frame(names=names, positions=positions, comment=comment, energy=None, forces=forces)


@register_io('xyz', 'write', 'xyz')
def write_frame_xyz(f_out, frame, unit=angstrom):
    """Print a single frame into an open XYZ file.

    This is currently hard-coded to write positions, if we ever need to write forces
    or something else, it needs generalizing.
    """

    # Check that required things are in frame:
    if (frame.positions is None) or (frame.names is None):
        raise ValueError('Frame does not contain required properties.')

    fmt_one = '{:13.6f}'
    fmt_prop = '{:6s} ' + 3*fmt_one + '\n'

    # write number of atoms and comment line
    f_out.write(f'{len(frame.names):d}\n')
    if frame.comment is not None:
        f_out.write(f'{frame.comment:s}\n')
    else:
        f_out.write('\n')

    data = frame.positions / unit

    # write atomic lines
    for i, name in enumerate(frame.names):
        f_out.write(fmt_prop.format(name, *data[i]))
