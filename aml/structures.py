"""Handling of atomic structures with all related information, including their properties."""

__all__ = [
    'Structure',
    'Structures',
    'Property',
    'Properties'
]

import fnmatch
import logging
from collections.abc import MutableMapping, MutableSequence
from itertools import islice
from typing import Optional, Sequence

import numpy as np

from .io import Frame, read_frames, write_frames
from .utilities import view_read_only


class Structure:
    """A single atomic structure.

    At atom names and positions have to be present, comment string and cell vectors are optional. All data is
    expected and assumed to be in atomic units.

    Data is hidden behind properties and returned as read-only views to prevent modification of values
    and/or shapes in place and thus protect data consistency. Everything but atom names can be modified by
    re-assignment, in which case it will go through the checks that are in place in the property setters.
    """

    __slots__ = '_names', '_positions', '_cell', '_comment', 'properties'

    _names: Sequence
    _positions: np.ndarray
    _cell: Optional[np.ndarray]
    _comment: Optional[str]
    _properties: Optional['Properties']

    @classmethod
    def from_frame(cls, frame):
        """Initialize with data from a frame dictionary.

        Arguments:
            frame: frame object, giving `None` as the default
        """
        if (frame.names is None) or (frame.positions is None):
            raise ValueError('Frame must have at least "names" and "positions".')
        return cls(frame.names, frame.positions, cell=frame.cell, comment=frame.comment)

    def __init__(self, names, positions, cell=None, comment=None, properties=None):
        """Initialize the atomic structure.

        The length of `names` determines the expected number of atoms and all checks are performed by property
        setters.

        Arguments:
            names: names of atoms
            positions: positions of atoms, array-like of dimensions (n_atoms, 3)
            cell: cell vectors, array-like of dimensions (3, 3)
            comment: comment string
            properties: `None` or an instance of `Properties`
        """

        self._set_names(names)
        self.positions = positions
        self.cell = cell
        self.comment = comment

        if properties is None:
            self.properties = Properties()
        else:
            if not isinstance(properties, Properties):
                raise TypeError('Expected a `Properties` instance for `properties`.')
            self.properties = properties

    @property
    def names(self):
        """Return the tuple of atom names directly - no modification possible."""
        return self._names

    def _set_names(self, names):
        """Set atomic names.

        This should not be called by the user after the object is constructed.
        """
        self._names = tuple(name for name in names)
        for name in self._names:
            if not isinstance(name, str):
                raise ValueError('`names` must be a sequence of strings.')

    @property
    def n_atoms(self):
        """The number of atoms is given by the number of names we have."""
        return len(self._names)

    @property
    def positions(self):
        """Show a read-only view to the user."""
        return view_read_only(self._positions)

    @positions.setter
    def positions(self, positions):
        """Set the positions, make sure it is an array of the right dimensions."""
        try:
            self._positions = np.array(positions, dtype=float)
            if self._positions.shape != (self.n_atoms, 3):
                raise ValueError
        except ValueError:
            raise ValueError('`positions` must be an array of shape (natoms, 3).')

    @property
    def cell(self):
        """Show a read-only view to the user."""
        return view_read_only(self._cell)

    @cell.setter
    def cell(self, cell):
        """Set the cell, make sure it is an array of the right dimensions."""
        if cell is not None:
            try:
                self._cell = np.array(cell, dtype=float)
                if self._cell.shape != (3, 3):
                    raise ValueError
            except ValueError:
                raise ValueError('`cell` must be a 3x3 array or `None`.')
        else:
            self._cell = None

    @property
    def comment(self):
        """Return the comment directly, as strings are not mutable."""
        return self._comment

    @comment.setter
    def comment(self, comment):
        """Set the comment string."""
        if (comment is not None) and (not isinstance(comment, str)):
            raise TypeError('`comment` must be a string or `None`.')
        else:
            self._comment = comment

    def to_frame(self, label_prop=None):
        """Return a frame corresponding to this structure.

        Arguments:
            label_prop: label of property to include, or `None`

        Returns:
            Frame
        """
        frame = Frame(names=self.names, positions=self.positions, cell=self.cell, comment=self.comment)
        # add stuff from a property if requested
        if label_prop is not None:
            prop = self.properties[label_prop]
            frame.energy = prop.energy
            frame.forces = prop.forces
        return frame

    def update_from_frame(self, frame, force: bool = False):
        """Update this structure with data from a frame."""

        # check that we have the same atom names
        if (frame.names is not None) and (self.names != tuple(frame.names)):
            raise ValueError('Inconsistent atom names.')

        # update all other stuff that is present
        attrs = ['positions', 'cell', 'comment']
        for attr in attrs:
            attr_frame = getattr(frame, attr)
            if attr_frame is not None:
                if force or (getattr(self, attr) is None):
                    setattr(self, attr, attr_frame)

    def __repr__(self):
        """Summarize the structure."""
        return 'Structure({:d} atoms)'.format(len(self._names))


class Structures(MutableSequence):

    @classmethod
    def from_file(cls, fn_in, fformat=None, **kwargs_from_frames):
        """Create structures from data in a file.

        The format of the file is given by `fformat` or inferred from the file
        extension if `fformat` is `None`. Remaining keyword arguments are passed to `from_frames`.

        Arguments:
            fn_in: name of the file
            fformat: format of the file, or `None`

        Returns:
            Structures
        """
        return cls.from_frames(read_frames(fn_in, fformat=fformat), **kwargs_from_frames)

    @classmethod
    def from_frames(cls, frames, start=0, stop=None, stride=1, probability=1.0, label_prop='reference'):
        """Create structures from frames.

        Here, each frame is an object that has all data to be included
        in that structure. The resulting `Structure` will simply be missing
        (i.e. have `None`) any data not given by the frame.

        Arguments:
            frames: an iterable over frame objects
            start: first frame to be considered
            stop: last frame to be considered
            stride: size of step through frames
            probability: take each considered frame with this probability
            label_prop: label under which to store any properties found in the frame

        Returns:
            Structures
        """

        structures = cls()
        for frame in islice(frames, start, stop, stride):
            # skip this frame if we don't hit the probability to be included
            if np.random.random() > probability:
                continue

            # construct one structure
            s = Structure.from_frame(frame)

            # if we are able to create a `Property` from the frame, store it
            property = Property.from_frame(frame)
            if property is not None:
                s.properties[label_prop] = property

            # store it
            structures.append(s)
        return structures

    def __init__(self, iterable=()):
        self._list = list()
        self.extend(iterable)

    def check(self, value):
        if not isinstance(value, Structure):
            raise TypeError('Can only contain `Structure` objects.')

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self._list[key])
        else:
            return self._list[key]

    def __delitem__(self, key):
        del self._list[key]

    def __setitem__(self, key, value):
        self.check(value)
        self._list[key] = value

    def __add__(self, other):
        structures = Structures(self)
        structures.extend(other)
        return structures

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __repr__(self):
        return '{:s}({:s})'.format(type(self).__name__, repr(list(self)))

    def insert(self, index, object):
        self.check(object)
        self._list.insert(index, object)

    def update_from_file(self, fn_in, fformat=None, **kwargs_from_frames):
        """Update structures and their properties from data in a file.

        The format of the file is given by `fformat` or inferred from the file
        extension if `fformat` is `None`. Remaining keyword arguments are passed
        to `update_from_frames`.

        Arguments:
            fn_in: name of the file
            fformat: format of the file, or `None`
        """
        return self.update_from_frames(read_frames(fn_in, fformat=fformat), **kwargs_from_frames)

    def update_from_frames(self, frames, label_prop='prediction', force: bool = False):
        """Updates structures and their properties from frames.

        Here, each frame will be used to update each structure in sequence.
        The property given by `label_prop` will also be updated (created, if needed), if possible.

        Arguments:
            frames: an iterable over frame objects
            label_prop: property label to set/update from the frame
            force: whether to overwrite data
        """

        for s, frame in zip(self, frames):

            # update the structure itself
            s.update_from_frame(frame, force=force)

            # update existing property or create it if needed
            if label_prop in s.properties.keys():
                s.properties[label_prop].update_from_frame(frame, force=force)
            else:
                property = Property.from_frame(frame)
                if property is not None:
                    s.properties[label_prop] = property

    def to_file(self, fn_out, fformat=None, **kwargs_to_frames):
        """Write all the structures to a file.

        The format of the file is given by `fformat` or inferred from the file
        extension if `fformat` is `None`. Remaining keyword arguments are passed to `to_frames`.

        Arguments:
            fn_out: name of the file
            fformat: format of the file, or `None`
        """
        write_frames(fn_out, self.to_frames(**kwargs_to_frames), fformat=fformat)

    def to_frames(self, label_prop=None):
        """Iterate over all structures as frames.

        Arguments:
            label_prop: label of property to include, or `None`
        """
        for s in self:
            yield s.to_frame(label_prop=label_prop)

    def sample(self, k):
        """Randomly sample structures without replacement.

        Randomly select `k` structures such that they do not repeat, i.e. perform
        sampling without replacement.

        Arguments:
            k: how many random samples to take

        Returns:
            a index array for new structures
        """
        idx = np.random.choice(len(self._list), k, replace=False)
        return list(idx)

    def sample_p(self, probability=1.0):
        """Sample the structures with a given probability.

        Arguments:
            probability: probability each structure will be included

        Returns:
            a index array for new structures
        """
        return list([i for i in range(len(self._list)) if probability > np.random.random()])

    def get(self, idx):
        """Return subset of original set based on index array"""
        return Structures(self._list[i] for i in idx)

    def reduce_property(self, f_reduce=np.std, name_prop='energy', label_prop='prediction*'):
        """Evaluate, for each structure, a function over multiple property labels.

        For each structure, defer to `Properties.reduce_property` to reduce its properties.
        All arguments as passed on to it, too. Return an array of these values for all structures.

        Arguments:
            f_reduce: function to evaluate over all labels
            name_prop: name of individual property
            label_prop: wildcard label of properties

        Returns:
            ndarray
        """

        results = []
        for s in self:
            data = s.properties.reduce_property(f_reduce=f_reduce, name_prop=name_prop, label_prop=label_prop)
            results.append(data)
        return np.array(results)

    def select_highest_error(self, n=10, label_prop='prediction*', f_reduce=np.std, name_prop='energy'):
        """Select n structures with highest score calculated based on reduce function
        for properties with shared label 'label_prop'.

        Arguments:
            n: number of selected structures
            label_prop: wildcard label of properties
            f_reduce: function to reduce properties
            name_prop: property to be used

        Returns:
            idx: Indices of selected structures
        """

        if n > len(self):
            raise ValueError('n larger than size of structures')

        # compute score by reducing label_prop property based on f_reduce function
        score = self.reduce_property(f_reduce=f_reduce, name_prop=name_prop, label_prop=label_prop)

        # select n highest
        idx = np.argpartition(score, -n)[-n:]
        logging.debug('Selected idx: {:}'.format(idx))
        logging.debug('Score of selected points: {:}'.format(score[idx]))
        logging.debug('Mean score of all points: {:g}'.format(score.mean()))

        # return indices
        return list(idx)

    def select_physical(self, cutoff=10, label_prop='prediction*', f_reduce=np.std, name_prop='energy'):
        """Select structures with score lower than cutoff, where the score is calculated
        based on given reduce function for properties with shared label 'label_prop'.

        Arguments:
            cutoff: threshold above which structures are ignored
            label_prop: wildcard label of properties
            f_reduce: function to reduce properties
            name_prop: property to be used

        Returns:
            idx: Indices of selected structures
        """

        # compute score by reducing label_prop property based on f_reduce function
        score = self.reduce_property(f_reduce=f_reduce, name_prop=name_prop, label_prop=label_prop)

        # select lower than cutoff
        idx = np.where(score < cutoff)[0]
        logging.debug('Selected idx: {:}'.format(idx))
        logging.debug('Score of selected points: {:}'.format(score[idx]))
        logging.debug('Mean score of all points: {:g}'.format(score.mean()))

        # generate and return new structures
        return list(idx)


class Property:
    """One or more properties of an atomic structure.

    Currently, the potential energy and forces are supported. Each property is optional, but at least one
    should be present. All data can also be changed later for an existing instance by re-assignment, in which
    case it will go through the checks that are in place in the property setters. Data is hidden behind
    properties and returned as read-only views to prevent modification of values and/or shapes in place and
    thus protect data consistency.

    All data is expected and assumed to be in atomic units.
    """

    __slots__ = '_energy', '_forces'

    @classmethod
    def from_frame(cls, frame):
        """Initialize with data from a Frame object.

        If no data relevant for properties is found in the frame, return `None`.

        Arguments:
            frame: default dataclass, giving `None` as the default

        Returns:
            Frame or `None`
        """

        # check that we have at least something, otherwise do not create anything
        if (frame.energy is None) and (frame.forces is None):
            return None
        else:
            return cls(energy=frame.energy, forces=frame.forces)

    def __init__(self, energy=None, forces=None):
        """Initialize the property with the given data.

        This will work even if all quantities are `None`, it is up to the user
        to make sure that some useful data is included.

        Arguments:
            energy: potential energy of the structure, one scalar
            forces: forces on atoms, array-like of dimensions (n_atoms, 3)
        """

        self.energy = energy
        self.forces = forces

    @property
    def energy(self):
        """Return the energy directly - no modification possible."""
        return self._energy

    @energy.setter
    def energy(self, energy):
        """Set the energy, make sure it is a float."""
        if energy is not None:
            try:
                self._energy = float(energy)
            except ValueError:
                raise ValueError('`energy` must be a float or `None`.')
        else:
            self._energy = None

    @property
    def forces(self):
        """Show a read-only view to the user."""
        return view_read_only(self._forces)

    @forces.setter
    def forces(self, forces):
        """Set the forces, make sure it is an array of the right dimensions."""
        if forces is not None:
            try:
                self._forces = np.array(forces, dtype=float)
                shape = self._forces.shape
                if (len(shape) != 2) or (shape[1] != 3):
                    raise ValueError
            except ValueError:
                raise ValueError('`forces` must be an array of shape (n_atoms, 3) or `None`.')
        else:
            self._forces = None

    def __repr__(self):
        ps = []
        if self.energy is not None:
            ps.append('energy')
        if self.forces is not None:
            ps.append('forces')
        return 'Property({:s})'.format(', '.join(ps))

    def update_from_frame(self, frame: Frame, force: bool = False):
        """Update with what data we can from a Frame object.

        Arguments:
            frame: a `Frame` object to get data from
            force: whether to overwrite data
        """

        attrs = ['energy', 'forces']
        for attr in attrs:
            attr_frame = getattr(frame, attr)
            if attr_frame is not None:
                if force or (getattr(self, attr) is None):
                    setattr(self, attr, attr_frame)


class Properties(MutableMapping):
    """A dictionary of `Property` objects.

    In addition to normal dictionary functionality, this also makes sure that values are only of type
    `Property` and provides the ability to glob keys. This make sense only with string keys,
    so let's stick to those.
    """

    def __init__(self, *args, **kwargs):
        self._dict = dict()
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        """Get one or more values from the dictionary.

        If `key` is present, the corresponding value is returned. If not, it is used as a globbing patern for
        keys and a list of the corresponding values is returned. If no keys match the pattern, a `KeyError` is
        raised, same as normal dictionary behavior.
        """

        # maybe we just find the key?
        try:
            return self._dict[key]
        except KeyError:
            pass

        # nope, let's try to glob it, then
        keys = self.keys_glob(key)
        if len(keys) == 0:
            raise KeyError("'{:s}'".format(key))
        return [self._dict[key] for key in keys]

    def __setitem__(self, key, value):
        """Store a value in the dictionary, with additional checks.

        We will not replace existing keys and only store `Property` objects.
        """

        # make sure we do not replace existing data
        if key in self.keys():
            raise KeyError('Label "{:s}" already present.'.format(key))

        # only store properties
        if not isinstance(value, Property):
            raise TypeError('Expected a `Property`.')

        # store the value
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '{:s}({:s})'.format(type(self).__name__, repr(self._dict))

    def keys_glob(self, pattern):
        """Find keys that match a pattern with shell-style wildcards.

        Returns:
            a list of matching keys, possibly empty

        Raises:
            TypeError, if any of the keys or the pattern are not strings
        """
        return fnmatch.filter(self.keys(), pattern)

    def reduce_property(self, f_reduce=np.std, name_prop='energy', label_prop='prediction*'):
        """Evaluate a function over multiple property labels.

        Select all properties given by a wildcard label and evaluate a function on them,
        usually a statistic, like a standard deviation.

        Arguments:
            f_reduce: function to evaluate over all labels
            name_prop: name of individual property
            label_prop: wildcard label of properties

        Returns:
            ndarray
        """

        try:
            data = np.array([getattr(p, name_prop) for p in self[label_prop]], dtype=np.float64)
        except KeyError:
            # the label was not found at all
            raise KeyError(f'Label "{label_prop}" not found in properties.')
        except TypeError:
            # the label was found, but only a single property
            raise TypeError(f'Label "{label_prop}" found only a single property.')
        except AttributeError:
            # label found, but the `Property` object does not have the requested attribute
            raise AttributeError(f'Property "{name_prop}" not present.')

        if np.isnan(data).any():
            msg = f'At least some values of "{name_prop}" were undefined (`None`) for label "{label_prop}".'
            raise ValueError(msg)

        return f_reduce(data)
