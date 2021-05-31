"""Functions to read and write n2p2 data files."""

__all__ = [
    'read_epre_n2p2',
    'read_fpre_n2p2'
]

import numpy as np

from .utilities import Frame, register_io


@register_io('N2P2_E','read')
def read_epre_n2p2(f_in, column=3):
    """Read the outcome of the energy prediction from file"""

    line = f_in.readline()
    # no more data in the file
    if not line:
        return None
    # Skip comment lines:
    while True:
        if '#' not in line:
            break
        line = f_in.readline()
    energy = float(line.split()[column])
    return Frame(energy=energy)


@register_io('N2P2_F','read')
def read_fpre_n2p2(f_in):
    """Read the outcome of the force prediction from file"""

    line = f_in.readline()
    # no more data in the file
    if not line:
        return None
    # Skip comment lines:
    while True:
        if '#' not in line:
            break
        line = f_in.readline()

    items = line.split()
    config = items[0]

    forces = []
    forces.append(float(items[3]))
    while True:
        last_pos = f_in.tell()
        line = f_in.readline()
        # no more data in the file
        if not line:
            break
        items = line.split()
        # Stop if config changes
        if items[0] != config:
            f_in.seek(last_pos)
            break

        forces.append(float(items[3]))

    forces = np.array(forces)
    forces = forces.reshape((len(forces)//3, 3))
    return Frame(forces=forces)
