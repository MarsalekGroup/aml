"""Functionality for handling atom centered symmetry functions.
"""

__all__ = [
    'RadialSF',
    'AngularSF',
    'generate_radial_shifted',
    'generate_angular_centered',
    'generate_radial_angular_default',
    'combine_radials_angulars_same',
    'format_ACSFs_radial',
    'format_ACSFs_angular',
    'format_combine_ACSFs',
]

from collections import Counter, namedtuple
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .constants import angstrom


# parameters of a radial ACSF
RadialSF = namedtuple('RadialSF', ['eta', 'mu', 'r_c'])


# parameters of an angular radial ACSF
AngularSF = namedtuple('AngularSF', ['lam', 'zeta', 'eta', 'r_c', 'mu'])


def cmp_same_count(x, y):
    """Comparison useful for excluding certain ACSFs."""
    return Counter(x) == Counter(y)


#
# Everything related to generating ACSFs
#


def generate_radial_shifted(n: int, r_0: float, r_max: float, r_c: float) -> List[RadialSF]:
    """Generate a systematic set of shifted radial ACSFs."""

    dr = (r_max - r_0) / (n - 1)
    eta = 1 / (2 * dr**2)

    radials = []
    for i in range(n):
        ri = r_0 + i * dr
        radials.append(RadialSF(eta=eta, mu=ri, r_c=r_c))

    return radials


def generate_angular_centered(n: int, r_0: float, r_max: float, zeta: float, r_c: float) -> List[AngularSF]:
    """Generate a systematic set of centered angular ACSFs."""

    if n % 2 != 0:
        raise ValueError('`n` must be even.')

    dr = (r_max - r_0) / (n - 1)

    angulars = []
    for i in range(n//2):
        mu = 0.0
        ri = r_0 + i * dr
        eta = 1 / (3 * ri**2)
        for lam in (-1, 1):
            angulars.append(AngularSF(lam=lam, zeta=zeta, mu=mu, eta=eta, r_c=r_c))

    return angulars


def generate_radial_angular_default() -> Tuple[List[RadialSF], List[AngularSF]]:
    """Generate a "default" set of radial and angular ACSFs.

    This is a convenience function to generate what we consider the "default" systematic set
    of 10 radial ACSFs and 4 angular ACSFs. All the parameters are hardcoded, if you want to
    customize this, use `generate_radial_shifted` and `generate_angular_centered` directly.
    """

    r_c = 12.0
    r_0_radial = 0.14 * angstrom
    r_max_radial = r_c - 0.14 * angstrom
    r_0_angular = 2.8 * angstrom
    r_max_angular = r_max_radial

    radials = generate_radial_shifted(n=10, r_0=r_0_radial, r_max=r_max_radial, r_c=r_c)

    angulars = (generate_angular_centered(n=2, r_0=r_0_angular, r_max=r_max_angular, zeta=1.0, r_c=r_c)
                + generate_angular_centered(n=2, r_0=r_0_angular, r_max=r_max_angular, zeta=4.0, r_c=r_c))

    return radials, angulars


def combine_radials_angulars_same(
    radials_pair: Sequence[RadialSF],
    angulars_triple: Sequence[AngularSF],
    elements: Sequence[str]
) -> Tuple[dict, dict]:
    """Combine radial and angular ACSFs for a set of elements, each pair/triple the same.

    Be mindful of the convention for elements 2 and 3 in angular ACSFs. Only one of the possible pairs is included,
    specifically one with index of element 3 same or higher than element 2.

    Arguments:
        radials_pair: radial ACSFs for a single pair
        angulars_triple: angular ACSFs for a single triple
        elements: a list of elements

    Returns:
        radials, angulars: each a dictionary keyed by tuples of elements
    """

    # dictionaries keyed by pairs and triples of elements, but here we use the same ACSFs for each pair/triple
    radials_elements = {}
    angulars_elements = {}

    # One would use the same loop in the general case, just create or provide different radial/angular ACSFs each time.
    for element1 in elements:
        for ie2, element2 in enumerate(elements):
            radials_elements[(element1, element2)] = radials_pair
            for ie3, element3 in enumerate(elements):
                if ie3 < ie2:
                    continue
                angulars_elements[(element1, element2, element3)] = angulars_triple

    return radials_elements, angulars_elements


#
# Everything related to string formatting ACSFs
#


def format_ACSFs_radial_single(radials: Sequence[RadialSF], element1: str, element2: Optional[str] = None) -> str:
    """Format (possibly weighted) radial ACSFs in input file format (RuNNer/n2p2).

    This formats and returns as a string radial ACSFs for a pair of elements (G2)
    or weighted radial ACSFs for a single element (G12) if the second element is not specified.

    G2 SF in n2p2 terminology. Documentation:
    https://compphysvienna.github.io/n2p2/api/symmetry_function_types.html#_CPPv4N3nnp12SymFncExpRadE

    G12 SF in n2p2 terminology. Documentation:
    https://compphysvienna.github.io/n2p2/api/symmetry_function_types.html#_CPPv4N3nnp20SymFncExpRadWeightedE
    """

    # prepare format line
    if element2 is None:
        # we're printing weighted ACSFs
        fmt_line = 'symfunction_short {element1:s} 12'
    else:
        # we're printing conventional ACSFs
        fmt_line = 'symfunction_short {element1:s} 2 {element2:s}'
    fmt_line += ' {eta:f} {mu:f} {r_c:f}'

    # format all the (w)ACSFs
    lines = []
    for radial in radials:
        lines.append(fmt_line.format(element1=element1, element2=element2, **radial._asdict()))

    # return the result as a multi-line string
    return '\n'.join(lines)


def format_ACSFs_radial(
    radials: dict,
    elements: Sequence[str],
    exclude_pairs: Sequence[Sequence[str]] = tuple(),
) -> str:
    """Format all radial ACSFs for a set of elements in input file format (RuNNer/n2p2).

    G2 SF in n2p2 terminology. Documentation:
    https://compphysvienna.github.io/n2p2/api/symmetry_function_types.html#_CPPv4N3nnp12SymFncExpRadE

    Arguments:
        radials: radial ACSFs dictionary keyed by tuples of two elements
        elements: a list of element names
        exclude_pairs: a list of element pairs to not consider (optional)
    """

    lines = []

    for element1 in elements:
        lines.append(f'#\n# Radial symmetry functions for {element1:s}\n#\n')
        for element2 in elements:
            comment = f'# {element1:s} - {element2:s}'
            if (exclude_pairs is not None):
                if any(cmp_same_count([element1, element2], exc) for exc in exclude_pairs):
                    lines.append(comment + ': excluded')
                    continue
            lines.extend([
                comment,
                format_ACSFs_radial_single(radials[(element1, element2)], element1, element2),
                ''])
        lines.append('')

    # return the result as a multi-line string
    return '\n'.join(lines)


def format_ACSFs_angular_single(
    angulars: Sequence[AngularSF],
    element1: str,
    element2: Optional[str] = None,
    element3: Optional[str] = None
) -> str:
    """Format (possibly weighted) angular ACSFs for a single triple of elements in input file format (RuNNer/n2p2).

    This formats angular ACSFs for a triple of elements (G3) or weighted angular ACSFs
    for a single element (G13) if the second and third elements are not specified.

    G3 SF in n2p2 terminology. Documentation:
    https://compphysvienna.github.io/n2p2/api/symmetry_function_types.html#_CPPv4N3nnp13SymFncExpAngnE

    G13 SF in n2p2 terminology. Documentation:
    https://compphysvienna.github.io/n2p2/api/symmetry_function_types.html#_CPPv4N3nnp21SymFncExpAngnWeightedE
    """

    # prepare format line
    if element2 is None:
        # we're printing weighted ACSFs
        if element3 is not None:
            raise ValueError('Specify both `element2` and `element3`, or neither.')
        fmt_line = 'symfunction_short {element1:s} 13'
    else:
        # we're printing conventional ACSFs
        fmt_line = 'symfunction_short {element1:s} 3 {element2:s} {element3:s}'
    fmt_line += ' {eta:f} {lam: f} {zeta:f} {r_c:f}'

    # format all the (w)ACSFs
    lines = []
    for angular in angulars:
        lines.append(fmt_line.format(element1=element1, element2=element2, element3=element3, **angular._asdict()))

    # return the result as a multi-line string
    return '\n'.join(lines)


def format_ACSFs_angular(
    angulars: dict,
    elements: Sequence[str],
    exclude_triples: Sequence[Sequence[str]] = tuple()
) -> str:
    """Format all angular ACSFs for a set of elements in input file format (RuNNer/n2p2).

    Be mindful of the convention for elements 2 and 3. Only one of the possible pairs is expected,
    in `angulars`, specifically one with index of element 3 same or higher than element 2.

    Arguments:
        angulars: angular ACSFs dictionary keyed by tuples of three elements
        elements: a list of element names
        exclude_triples: a list of element triples to not consider (optional)
    """

    lines = []

    for element1 in elements:
        lines.append(f'#\n# Angular symmetry functions for {element1:s}\n#\n')
        for ie2, element2 in enumerate(elements):
            for ie3, element3 in enumerate(elements):
                if ie3 < ie2:
                    continue
                comment = f'# {element1:s} - {element2:s}-{element3:s}'
                if (exclude_triples is not None):
                    if any(cmp_same_count([element1, element2, element3], exc) for exc in exclude_triples):
                        lines.append(comment + ': excluded')
                        continue
                lines.extend([
                    comment,
                    format_ACSFs_angular_single(angulars[(element1, element2, element3)], element1, element2, element3),
                    ''])
        lines.append('')

    # return the result as a multi-line string
    return '\n'.join(lines)


def format_combine_ACSFs(
    radials: Sequence[RadialSF],
    angulars: Sequence[AngularSF],
    elements: Sequence[str],
    exclude_pairs: Sequence[Sequence[str]] = tuple(),
    exclude_triples: Sequence[Sequence[str]] = tuple()
) -> str:
    """Combine radial and angular ACSFs for a set of elements and format them in input file format (RuNNer/n2p2).

    Format all the radial ACSFs for each pair of elements in `elements`
    and all the angular ACSFs for each triple of elements in `elements`.

    Arguments:
        radials: radial symmetry functions
        angulars: angular symmetry functions
        elements: a list of element names
        exclude_pairs: a list of element pairs to not consider (optional)
        exclude_triples: a list of element triples to not consider (optional)
    """

    radials_elements, angulars_elements = combine_radials_angulars_same(radials, angulars, elements)

    # string with formatted radial symmetry functions
    radials_str = format_ACSFs_radial(radials_elements, elements, exclude_pairs)

    # string with formatted angular symmetry functions
    angulars_str = format_ACSFs_angular(angulars_elements, elements, exclude_triples)

    return radials_str + '\n' + angulars_str


#
# Everything related to plotting ACSFs
#


def f_cut_cos(r, r_c):
    """Cosine type cutoff function."""
    return (np.heaviside(r_c-r, 1)) * 0.5 * (np.cos(np.pi*r/r_c) + 1)


def f_radial(r, r_c, mu, eta, f_cut):
    """Radial symmetry function.

    This is the same for G2 and G12.
    """
    return f_cut(r, r_c) * np.exp(- eta * (r-mu)**2)


def f_angular(theta, angular):
    """Purely angular part of angular symmetry functions.

    This is the same for G3, G9, G13.
    """
    return 2**(1-angular.zeta) * (1 + angular.lam * np.cos(theta))**angular.zeta


def plot_radial(radials, f_cut, normalize=True, filename=None):
    """Plot radial SFs or radial dependence of angular SFs.

    Arguments:
        radials: a sequence of `RadialSF`s
        f_cut: a cutoff function
        normalize: whether to normalize to max height = 1
        filename: name of file to save plot, optional
    """

    plt.figure(constrained_layout=True)

    # find all possible cutoffs and plot cutoff functions
    r_cs = {radial.r_c for radial in radials}
    r = np.linspace(0, max(r_cs), 1000)
    for r_c in r_cs:
        plt.plot(r*angstrom, f_cut(r, r_c), linestyle='dotted', color='gray', lw=2)

    # find all possible combinations of (r_c, mu, eta) and plot the SFs
    params_all = {(radial.r_c, radial.mu, radial.eta) for radial in radials}
    for r_c, mu, eta in params_all:
        f = f_radial(r, r_c, mu, eta, f_cut)
        if normalize:
            f /= max(f)
        plt.plot(r*angstrom, f, color='darkblue', lw=2)

    # wrap up
    plt.xlabel('Radial distance [Å]')
    plt.ylabel('ACSF value')

    if filename is not None:
        plt.savefig(filename)


def plot_angular(angulars, filename=None):
    """Plot angular part of angular SFs.

    Arguments:
        angulars: a sequence of `AngularSF`s
        filename: name of file to save plot, optional
    """

    theta = np.linspace(0, 2*np.pi, 1000)

    plt.figure(constrained_layout=True)

    for angular in angulars:
        plt.plot(theta, f_angular(theta, angular), lw=2)

    # wrap up
    plt.xlabel('θ [rad]')
    plt.ylabel('ACSF value')
    plt.xticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])

    if filename is not None:
        plt.savefig(filename)
