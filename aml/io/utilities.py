"""Input and output utilities.

The central concept in the I/O infrastructure is a "frame" - a dataclass that represents one atomic
configuration that contains data of different kinds. Atomic units are used in the frame object itself,
unless explicitly stated otherwise. Units specified by the file format are used in the files themselves.
"""

__all__ = [
    'AnyPath',
    'get_fn_test',
    'Frame',
    'open_safe',
    'working_directory',
    'temporary_directory',
    'to_file',
    'from_file',
    'read_frames',
    'write_frames',
    'merge_frames',
]

import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ..utilities import AMLIOError


# functions that are registered to read and write frames
formats = defaultdict(dict)


# mapping of file extensions to file formats
ext2fmt = dict()


AnyPath = Union[str, Path]


def get_fn_test(filename):
    """Get absolute file names of test data.

    Arguments:
        filename: name of file in the test data directory, no path
    """
    fn_out = Path(__file__).parent.parent / '../tests/data' / filename
    return fn_out.resolve()


def register_io(fformat: str, operation: str, extension: Union[str, None] = None):
    """Decorator to register an I/O operation for a specific file format.

    Optionally, the function can also register a file name extension to automatic
    detection of file format from file name.

    Arguments:
        fformat: name of file format
        operation: I/O operation - "read" or "write"
        extension: file name extension or `None`
    """
    def decorator(function):
        if operation not in ('read', 'write'):
            raise ValueError('Unrecognized operation. Allowed values: "read", "write".')
        formats[fformat][operation] = function
        if extension is not None:
            formats[fformat]['extension'] = extension
            if (extension in ext2fmt.keys()) and ext2fmt[extension] != fformat:
                raise ValueError(f'Attempted to register the same file extension ({extension}) twice.')
            ext2fmt[extension] = fformat
    return decorator


@dataclass(eq=False)
class Frame:
    """All possible data of a single frame.

    Used to exchange data between data structure and I/O routines. Defaults are set to `None`, which
    corresponds to that given kind of data not being set/available. We do not provide a comparison operator,
    at least for now, as comparing NumPy arrays is more involved.
    """

    # slots do not work correctly with dataclass
    # Here is an alternative: https://pypi.org/project/dataslots/
    # Here is some context: https://github.com/ericvsmith/dataclasses/issues/28
    # __slots__ = ['names', 'positions', 'cell', 'comment', 'energy', 'forces']

    names: Optional[Sequence] = None
    positions: Optional[np.ndarray] = None
    cell: Optional[np.ndarray] = None
    comment: Optional[str] = None
    energy: Optional[float] = None
    forces: Optional[np.ndarray] = None

    def update(self, other: 'Frame', force: bool = False):
        """Update this frame with data from another.

        Arguments:
            other: another frame
            force: whether to overwrite data
        """

        # check that we have the same atom names
        if (other.names is not None) and (self.names != other.names):
            raise ValueError('Inconsistent atom names.')

        # take over all that we can
        attrs = ['positions', 'cell', 'comment', 'energy', 'forces']
        for attr in attrs:
            attr_o = getattr(other, attr)
            if attr_o is not None:
                if force or (getattr(self, attr) is None):
                    setattr(self, attr, attr_o)


def open_safe(filename, mode='r', buffering=-1, verbose=False):
    """A wrapper around `open` which saves backup files.

    If opening for writing and `filename` exists, it will be renamed
    so that we do not overwrite any data.

    Arguments:
        filename: name of file to open
        mode: file open mode
        buffering: passed through to `open`
        verbose: whether to print to standard output what backup was performed

    Returns:
        an open file
    """

    if mode[0] == 'w':
        # if writing, make sure file is not overwritten

        filename = Path(filename)

        i = 0
        fn_backup = filename
        while fn_backup.exists():
            name_new = f'#{filename.name:s}#{i:d}#'
            fn_backup = fn_backup.with_name(name_new)
            i += 1

        if fn_backup != filename:
            filename.rename(fn_backup)
            if verbose:
                print(f'Backup performed: {filename} -> {fn_backup}\n')

    elif mode[0] in ('r', 'a'):
        # read or append, no danger of overwritten files
        pass

    else:
        # did not expect that, more work needed
        raise NotImplementedError(f'Unsupported file open mode: {mode:s}.')

    return open(filename, mode, buffering)


@contextmanager
def working_directory(directory):
    """Change working directory within the context.

    This is not available in the standard library [1] but can be useful, especially for testing.
    The old fixture in pytest (`tmpdir`) used py.path [2] which has `as_cwd`, but this is legacy
    code now and not recommended [3].

    [1] https://bugs.python.org/issue25625
    [2] https://py.readthedocs.io/en/latest/path.html
    [3] https://docs.pytest.org/en/latest/how-to/tmpdir.html

    Arguments:
        directory: directory to change to
    """

    # store the current working directory
    dir_original = Path().absolute()

    # try to change to the new one and then back
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(dir_original)


@contextmanager
def temporary_directory(directory: AnyPath, parents: bool = False, keep: bool = False):
    """Create a temporary directory.

    The directory is removed upon exiting the context, unless the users asks to keep it.

    Arguments:
        directory: directory to create
        parents: whether to create parents as well
        keep: whether to keep directory after exiting context
    """

    directory = Path(directory)

    # catch conflict early, a nicer error message
    if directory.exists():
        raise AMLIOError(f'Unable to create directory, already exists: {directory.absolute()}')

    # actually make the directory
    directory.mkdir(parents=parents)

    # create context, clean up if needed
    try:
        yield directory
    finally:
        if not keep:
            shutil.rmtree(directory)


def from_file(fn_in, binary=False):
    """Read the contents of a file into a variable.

    By default, the file will be read as a text file, resulting in a string.
    It `binary` is true, it will be read as a binary file, resulting in bytes.
    """

    mode = 'r'
    if binary:
        mode += 'b'
    with open(fn_in, mode) as f_in:
        data = f_in.read()
    return data


def to_file(data, fn_out, binary=False, verbose=False):
    """Write a variable to a file.

    The provided `data` would typically be a string or bytes, if `binary` is true.
    The output file name is protected against overwriting and if `verbose is true,
    backup file creation will be reported.
    """

    mode = 'w'
    if binary:
        mode += 'b'
    with open_safe(fn_out, mode, verbose=verbose) as f_out:
        f_out.write(data)


def get_io_operation(fn, fformat, operation):
    """Select I/O function for given file format.

    Arguments:
        fn: name of file to operate on
        fformat: name of file format
        operation: I/O operation - "read" or "write"

    Returns:
        function to read or write one frame
    """

    if operation not in ('read', 'write'):
        raise ValueError('Unrecognized operation. Allowed values: "read", "write".')

    # automatically pick a file format
    if fformat is None:
        fn = Path(fn)
        extension = fn.suffix[1:]
        try:
            fformat = ext2fmt[extension]
        except KeyError:
            raise KeyError(f'Extension "{extension:s}" not registered for file format detection.')

    try:
        return formats[fformat][operation]
    except KeyError:
        msg = f'File format "{fformat:s}" not supported for operation "{operation:s}".'
        raise ValueError(msg)


def read_frames(fn_in, fformat=None, **kwargs):
    """Iterate over a trajectory file, returning all data for each frame."""

    read_frame = get_io_operation(fn_in, fformat, 'read')

    # read all frames, quit when there is no more data
    # File formats read using MDTraj must be opened differently. Maybe there is a more elegany way to do that though
    with open(fn_in) as f_in:
        while True:
            frame = read_frame(f_in, **kwargs)
            if frame is None:
                break
            yield frame


def write_frames(fn_out, frames, fformat=None):
    """Write frames to file.

    The format of the file is given by `fformat` or inferred from the file
    extension if `fformat` is `None`.

    Arguments:
        fn_out: name of output file
        frames: iterator over `Frame` objects
        fformat: format of the file, or `None`
        label_prop: label of property to include, or `None`
    """

    write_frame = get_io_operation(fn_out, fformat, 'write')

    # write all frames to file
    with open_safe(fn_out, 'w') as f_out:
        for frame in frames:
            write_frame(f_out, frame)


def merge_frames(frames, *frames_others, force: bool = False):
    """Merge frames from multiple sources.

    The length of the result will be determined by the length of `frames`,
    the other iterators should be at least as long as that.

    Arguments:
        frames: iterator over `Frame` objects
        frames_others: more iterators over `Frame` instances
        force: whether to overwrite data

    Yields:
        `Frame` objects
    """

    for frame in frames:
        for frames_extra in frames_others:
            frame.update(next(frames_extra), force=force)
        yield frame
