from copy import copy
from functools import partial

import numpy as np
import pytest

from aml import Structure, Structures, Property, Properties, Frame


def test_Structure_init():

    names = ['H', 'H']
    positions = [[0, 0, 0], [1, 0, 0]]
    forces = [[0, 0, 0], [0, 0, 0]]
    cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    properties = Properties(reference=Property(energy=0.0, forces=forces))
    frame = Frame(names=names, positions=positions)

    Structure(['H'], [[0, 0, 0]])

    s = Structure(names, positions)
    s.names
    s.positions
    s.cell
    s.comment

    s = Structure(names, positions, cell=cell, comment='some atoms')
    s.names
    s.positions
    s.cell
    s.comment

    # construct with properties
    Structure(names, positions, cell=cell, comment='some atoms', properties=properties)

    p = Property(energy=42, forces=forces)
    s.properties['test'] = p
    s.properties['test'].energy
    s.properties['test'].forces

    s.positions = positions
    s.cell = cell
    s.comment = 'other atoms'
    s.properties['test'].energy = 42
    s.properties['test'].forces = forces

    with pytest.raises(ValueError, match='`names` must be a sequence of strings.'):
        Structure(['H', 1], positions)

    # names can't be changed
    with pytest.raises(TypeError, match="'tuple' object does not support item assignment"):
        s.names[0] = 'abc'

    # names can't be reassigned
    with pytest.raises(AttributeError, match="can't set attribute"):
        s.names = ['a', 'b']

    # positions can't be changed (can be reassigned)
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        s.positions[0, 0] = 42

    # correct dimensions of positions
    with pytest.raises(ValueError, match='`positions` must be an array of shape .natoms, 3..'):
        s.positions = np.zeros(3)

    # cell can't be changed (can be reassigned)
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        s.cell[0, 0] = 42

    # correct dimensions of cell
    with pytest.raises(ValueError, match='`cell` must be a 3x3 array or `None`.'):
        s.cell = np.zeros(3)

    # only pass `Properties`
    with pytest.raises(TypeError, match='Expected a `Properties` instance for `properties`.'):
        Structure(names, positions, cell=cell, comment='some atoms', properties=[1, 2, 3])

    # required data not present in frame
    with pytest.raises(ValueError, match='Frame must have at least "names" and "positions".'):
        Structure.from_frame(Frame())

    # comment must be a string
    with pytest.raises(TypeError, match='`comment` must be a string or `None`.'):
        Structure(names, positions, comment=42)

    # construct from a frame
    s2 = Structure.from_frame(frame)

    # update from frame with the same frame just runs
    s2.update_from_frame(frame)

    assert s2
    s2.update_from_frame(Frame(comment='new comment'))

    # can't update from a frame with mismatched atom names
    with pytest.raises(ValueError, match='Inconsistent atom names.'):
        s2.update_from_frame(Frame(names=['H'], positions=positions))


def test_Structures_init():

    names = ['H', 'H']
    positions = [[0, 0, 0], [1, 0, 0]]
    s = Structure(names, positions)
    frame = Frame(names=names, positions=positions)

    Structures()

    structures = Structures(s for i in range(3))
    assert type(structures) is Structures

    structures = Structures([s, s])
    assert type(structures) is Structures

    # construct from frames with only basic data
    frames = [frame, frame]
    Structures.from_frames(frames)
    np.random.seed(42)
    Structures.from_frames(frames, probability=0.01)
    Structures.from_frames(frames, stride=2)


def test_Structures():

    names = ['H', 'H']
    positions = [[0, 0, 0], [1, 0, 0]]
    s = Structure(names, positions)

    # slicing
    structures = Structures(s for i in range(3))
    assert type(structures[1:3]) is Structures
    assert type(structures[0]) is Structure

    # + operator
    structures_1 = Structures(s for i in range(3))
    structures_2 = Structures(s for i in range(2))
    structures_both = structures_1 + structures_2
    assert type(structures_both) is Structures
    assert len(structures_both) == len(structures_1) + len(structures_2)

    # + operator with list
    structures_new = structures_1 + [s, s]
    assert type(structures_new) is Structures
    assert len(structures_new) == len(structures_1) + 2

    # += operator
    structures_1 += structures_2
    assert len(structures_1) == 3 + 2

    # random sampling
    assert type(structures_1.get(structures_1.sample(5))) is Structures
    assert type(structures_1.get(structures_1.sample_p(0.5))) is Structures

    # __setitem__
    structures[0] = s

    # item deletion
    structures.pop()

    # text representation
    repr(structures)

    # can only contain `Structure` objects
    with pytest.raises(TypeError, match='Can only contain `Structure` objects.'):
        structures.append(42)


def test_Property():

    energy = 0.0
    forces = [[0.0, 0.0, 0.0]]

    Property()
    Property(energy=energy)
    Property(forces=forces)

    with pytest.raises(ValueError, match='`forces` must be an array of shape .n_atoms, 3. or `None`.'):
        Property(forces=0.0)

    with pytest.raises(ValueError, match='`energy` must be a float or `None`.'):
        Property(energy='a')

    # attribute access
    p = Property(energy=energy, forces=forces)
    p.energy
    p.forces

    # text representation
    repr(p)


def test_Properties():

    p = Property()

    properties = Properties()
    properties['test-1'] = p
    properties['test-2'] = p
    list(properties.keys())
    assert type(properties['test-1']) is Property
    assert type(properties['test-*']) is list

    # text representation
    repr(properties)

    del properties['test-2']

    with pytest.raises(KeyError, match='"\'nope\'"'):
        properties['nope']

    with pytest.raises(KeyError, match='Label "test-1" already present.'):
        properties['test-1'] = p

    with pytest.raises(TypeError, match='Expected a `Property`.'):
        properties['new'] = 'value'


def test_sample():
    names = ['H', 'H']
    positions = [[0, 0, 0], [1, 0, 0]]
    s = Structure(names, positions)

    structures = Structures(s for i in range(10))

    s_sample = structures.get(structures.sample(5))
    assert type(s_sample) is Structures
    assert len(s_sample) == 5

    s_sample = structures.get(structures.sample_p(0.5))
    assert type(s_sample) is Structures


def get_structures_properties():

    names = ['H', 'H']
    positions = [[0, 0, 0], [1, 0, 0]]
    prop = Property(energy=0.0, forces=[[0, 0, 0], [0, 0, 0]])
    properties = Properties(one=copy(prop), two=copy(prop))
    s1 = Structure(names, positions, properties=copy(properties))
    s2 = Structure(names, positions, properties=copy(properties))
    s2.properties['one'].energy = 1.0

    return Structures((s1, s2))


def test_reduce():

    structures = get_structures_properties()

    std = structures.reduce_property(label_prop='*')
    assert type(std) is np.ndarray
    assert std.shape == (2,)

    mean_force = structures.reduce_property(
        f_reduce=partial(np.mean, axis=0),
        name_prop='forces',
        label_prop='*'
    )
    assert type(mean_force) is np.ndarray
    assert mean_force.shape == (2,2,3)

    # property label is not present
    with pytest.raises(KeyError, match='Label "nope" not found in properties.'):
        structures.reduce_property(label_prop='nope')

    # property of this name is not present at all
    with pytest.raises(AttributeError, match='Property "nope" not present.'):
        structures.reduce_property(name_prop='nope', label_prop='*')

    # property label found, but only a single one - nothing to reduce
    with pytest.raises(TypeError, match='Label "one" found only a single property.'):
        structures.reduce_property(label_prop='one')

    # property of this name is not present for one of the structures
    structures[0].properties['one'].energy = None
    with pytest.raises(ValueError):
        structures.reduce_property(name_prop='energy', label_prop='*')


def test_select():

    structures = get_structures_properties()

    s_sel = structures.get(structures.select_highest_error(n=1, label_prop='*'))
    assert type(s_sel) is Structures
    assert len(s_sel) == 1

    with pytest.raises(ValueError, match='n larger than size of structures'):
        structures.select_highest_error(n=3, label_prop='*')

    with pytest.raises(KeyError, match='Label "nope" not found in properties.'):
        structures.select_highest_error(1, label_prop='nope')

    with pytest.raises(TypeError, match='Label "one" found only a single property.'):
        structures.select_highest_error(1, label_prop='one')

    structures[0].properties['one'].energy = None
    with pytest.raises(ValueError):
        structures.select_highest_error(1, label_prop='*')
