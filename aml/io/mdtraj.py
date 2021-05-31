"""Read data using MDTraj."""

__all__ = [
    'read_frames_mdtraj',
]

try:
    import mdtraj
except ImportError:
    mdtraj = None

from .utilities import Frame
from ..constants import nm


def read_frames_mdtraj(fn_in, top=None, names_atoms='type', name_data='positions', unit=nm, unit_cell=nm, chunk=100):
    """Read data from a file using the MDTraj package.

    Arguments:
        fn_in: name of trajectory file to read, passed to `mdtraj.iterload`
        top: MDTraj topology, passed to `mdtraj.iterload`
        names_atoms: which atom names to use, 'type' or 'element'
        name_data: what quantity to take the data as
        unit: unit to scale data by, multiplicative factor in atomic units
        unit_cell: unit to scale cell by, multiplicative factor in atomic units
        chunk: size of one trajectory chunk, passed to `mdtraj.iterload`

    Yields:
        One AML `Frame` object at a time
    """

    # open the trajectory for interation
    trj = mdtraj.iterload(fn_in, top=top, chunk=chunk)

    # no atom names yet
    names = None

    # prepare data names
    if name_data not in ('positions', 'forces'):
        raise ValueError(f'Unsupported `name_data`: {name_data}. Expected "positions" or "forces".')

    # iterate over all frames
    for chunk in trj:

        # prepare atom names
        # (`trj` is a generator, no topology information there)
        if names is None:
            if names_atoms == 'type':
                names = [atom.name for atom in chunk.topology.atoms]
            elif names_atoms == 'element':
                names = [atom.element.symbol for atom in chunk.topology.atoms]
            else:
                raise ValueError(f'Expected "type" or "element" for `name_atoms`, got {names_atoms}.')

        for i in range(len(chunk)):

            # atomic data
            data = chunk.xyz[i, :, :] * unit

            # cell data, if present
            if chunk.unitcell_vectors is not None:
                cell = chunk.unitcell_vectors[i, ...] * unit_cell
            else:
                cell = None

            # prepare all kwargs and construct a frame
            kwargs = {
                'names': names,
                name_data: data,
                'cell': cell
            }
            yield Frame(**kwargs)


if mdtraj is None:
    del read_frames_mdtraj
    __all__.remove('read_frames_mdtraj')
