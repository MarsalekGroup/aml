import difflib

from aml.acsf import (AngularSF, RadialSF, combine_radials_angulars_same, format_combine_ACSFs,
                      generate_radial_angular_default)
from aml.io import get_fn_test


def print_str_diff(left, right):

    print('LEFT: START')
    print(left)
    print('LEFT: END')
    print()

    print('RIGHT: START')
    print(right)
    print('RIGHT: END')
    print()

    diff = difflib.ndiff(left.splitlines(), right.splitlines())
    for line in diff:
        print(line)
    print()


def test_acsf_default():

    # all default ACSFs
    radials, angulars = generate_radial_angular_default()

    # expected output
    assert all([type(r) == RadialSF for r in radials])
    assert all([type(a) == AngularSF for a in angulars])
    assert len(radials) == 10
    assert len(angulars) == 4

    # create dictionaries for specific elements
    elements = ['H', 'O']
    radials_elements, angulars_elements = combine_radials_angulars_same(radials, angulars, elements)

    # perform some checks
    assert len(radials_elements.keys()) == 4
    assert len(angulars_elements.keys()) == 6
    assert len(radials_elements[('H', 'H')]) == 10

    # format n2p2 input lines
    acsf_str = format_combine_ACSFs(radials, angulars, elements)

    # load the expected formatted lines
    acsf_str_ref = open(get_fn_test('acsf-default-H-O-all.txt')).read()

    # if they don't match, print extra information that will show up when the assert fails
    if acsf_str != acsf_str_ref:
        print_str_diff(acsf_str_ref, acsf_str)

    # check that the formatted lines match
    assert acsf_str == acsf_str_ref


def test_acsf_exclusions():

    # all default ACSFs
    radials, angulars = generate_radial_angular_default()

    # format n2p2 input lines with some exclusions
    acsf_str = format_combine_ACSFs(
        radials,
        angulars,
        elements=['H', 'O'],
        exclude_pairs=[['O', 'O']],
        exclude_triples=[['O', 'O', 'O'], ['H', 'O', 'O']]
    )

    # load the expected formatted lines
    acsf_str_ref = open(get_fn_test('acsf-default-H-O-exclude-O-O.txt')).read()

    # if they don't match, print extra information that will show up when the assert fails
    if acsf_str != acsf_str_ref:
        print_str_diff(acsf_str_ref, acsf_str)

    # check that the formatted lines match
    assert acsf_str == acsf_str_ref
