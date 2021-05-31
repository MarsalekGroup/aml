"""Functions to read and write RuNNer data files."""

__all__ = [
    'write_frame_runner',
    'read_frame_runner',
]

import numpy as np

from .utilities import Frame, register_io


@register_io('RuNNer', 'read', 'data')   # noqa: C901
def read_frame_runner(f_in):
    """Read one frame of the RuNNer format from an open file.

    Arguments:
        f_in: open file in the RuNNer format

    Returns:
        `Frame` instance or `None`
    """

    # For reference, in n2p2, this is implemented in `Structure::readFromFile`, found somewhere here:
    # https://github.com/CompPhysVienna/n2p2/blob/master/src/libnnp/Structure.cpp#L84

    # read first line to examine it
    line_begin = f_in.readline()

    # no more data in the file
    if not line_begin:
        return None

    # there is some data, frame should start with 'begin'
    if line_begin.strip() != 'begin':
        raise ValueError

    comment = None
    cell = []
    names = []
    positions = []
    forces = []
    energy = None

    for line in f_in:
        items = line.split()
        tag = items[0]

        if tag == 'comment':
            comment = " ".join(items[1:])

        elif tag == 'lattice':
            cell.append([float(item) for item in items[1:]])

        elif tag == 'atom':
            positions.append([float(item) for item in items[1:4]])
            names.append(items[4])
            forces.append([float(item) for item in items[7:10]])
            # items[5] is atomic energy, only RuNNer itself (potentially) deals with that
            # items[6] is atomic energy - not really used by anyone

        elif tag == 'energy':
            energy = float(items[1])

        elif tag == 'charge':
            pass

        elif tag == 'end':
            break

        else:
            raise ValueError('Unexpected data in file.')

    if len(names) == 0:
        raise ValueError('No atomic data.')
    cell = np.array(cell)
    if cell.shape != (3, 3) and len(cell) != 0:
        raise ValueError('Wrong cell data.')
    if len(cell) == 0:
        cell = None
    positions = np.array(positions)
    forces = np.array(forces)

    # Prepare frame
    frame = Frame(names=names, positions=positions, comment=comment, cell=cell, energy=energy, forces=forces)

    return frame


@register_io('RuNNer', 'write', 'data')
def write_frame_runner(f_out, frame):

    # "cell" and "lattice" is the same data, we just use the terminology of the file format here.
    #
    # Note that atomic charges, atomic energies, and total charge currently not supported
    # and zeros will be written in the file for these.

    # Check that required data is in the frame:
    if (frame.positions is None) or (frame.names is None):
        raise ValueError('Frame does not contain required properties - atom names and positions.')

    fmt_lattice = 'lattice ' + 3*'{:16.6f}' + '\n'
    fmt_one = '{:13.6f}'
    fmt_atom = 'atom ' + 3*fmt_one + '{:^6s}' + 5*fmt_one + '\n'
    fmt_energy = 'energy ' + fmt_one + '\n'
    fmt_charge = 'charge ' + fmt_one + '\n'

    f_out.write('begin\n')

    if frame.comment is not None:
        f_out.write('comment ' + frame.comment + '\n')

    if frame.cell is not None:
        for lattice_vector in frame.cell:
            f_out.write(fmt_lattice.format(*lattice_vector))

    if frame.forces is not None:
        for i, name in enumerate(frame.names):
            f_out.write(fmt_atom.format(*frame.positions[i], name,
                                        0.0, 0.0, *frame.forces[i]))
    else:
        for i, name in enumerate(frame.names):
            f_out.write(fmt_atom.format(*frame.positions[i], name,
                                        0.0, 0.0, 0.0, 0.0, 0.0))

    if frame.energy is None:
        energy = 0.0
    else:
        energy = frame.energy
    f_out.write(fmt_energy.format(energy))

    f_out.write(fmt_charge.format(0.0))

    f_out.write('end\n')
