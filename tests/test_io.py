from aml import Structures
from aml.io import from_file, to_file, working_directory, get_fn_test


def test_from_to_file(tmp_path):
    """Test reading data from files to variables and writing data in variables to files."""

    with working_directory(tmp_path):

        data = from_file(get_fn_test('h9o4p.xyz'))
        to_file(data, 'out.xyz')

        data_b = from_file(get_fn_test('h9o4p.xyz'), binary=True)
        to_file(data_b, 'out-binary.xyz', binary=True)


def test_xyz(tmp_path):
    """Test file input/output for xyz."""

    # get the names of some test files - from CP2K and written by this code
    fn_orig = get_fn_test('h9o4p.xyz')
    fn_ours = get_fn_test('h9o4p_ours.xyz')

    # binary read of our reference
    data_ref = open(fn_ours, 'rb').read()

    # read the original
    s = Structures.from_file(fn_orig)

    # in a temporary directory...
    with working_directory(tmp_path):

        # write it...
        fn_written = 'test.xyz'
        s.to_file(fn_written)

        # ... and compare byte-for-byte with the stored reference
        data_written = open(fn_written, 'rb').read()
        assert data_written == data_ref


def test_runner(tmp_path):
    """Test file input/output for RuNNer."""

    # get the names of some test files - from CP2K and written by this code
    fn_orig = get_fn_test('water.data')
    fn_ours = get_fn_test('water_ours.data')

    # binary read of our reference
    data_ref = open(fn_ours, 'rb').read()

    # read the original
    s = Structures.from_file(fn_orig)

    # in a temporary directory...
    with working_directory(tmp_path):

        # write it...
        fn_written = 'test.data'
        s.to_file(fn_written, label_prop='reference')

        # ... and compare byte-for-byte with the stored reference
        data_written = open(fn_written, 'rb').read()
        assert data_written == data_ref
