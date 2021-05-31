"""Shared utility functions."""

__all__ = [
    "plot_all_f_and_errors",
    "plot_mae_errors",
    "print_errors"
]

import numpy as np
import matplotlib.pyplot as plt


def get_unique_atom_types(topology):
    """Determine the unique atom types in a trajectory."""

    atom_types = list(set(atom.name for atom in topology.atoms))

    return atom_types


def get_unique_elements(topology):
    """Determine the unique elements in a trajectory."""

    elements = list(set(atom.element.symbol for atom in topology.atoms))

    return elements


def compute_all_errors(ref_fnc, test_fnc):
    """Compute MAEs relative to reference data.

    Given two sets of functions returns the mean absolute error
    of the test functions relative to the reference (first argument).
    """

    error = {}
    for name, data in ref_fnc.items():
        # Rescale array in case of different resolution
        test_data = np.interp(data[0], test_fnc[name][0], test_fnc[name][1])
        diff = data[1] - test_data
        mae = np.sum(np.absolute(diff)) / (np.sum(data[1]) + np.sum(test_data))
        error[name] = [data[0], diff, mae]
    return error


def print_errors(error):
    """Function to print the errors in a human readable way."""

    n = 22
    print(n*"=")
    print(f"{'Score Summary':{n}s}")
    print(n*"=")
    print("Label   | Accuracy [%]")
    print(n*"_")

    all_err = []
    for name, data in error.items():
        err = (1 - data[2]) * 100
        print(f"{name:7s} | {err:3.4}")
        all_err.append(err)

    # Mean of errors
    name = 'Mean'
    err = np.mean(all_err)
    print(f"{name:7s} | {err:3.4}")

    print(n*"=")


def plot_all_f_and_errors(ref, test, error, observable):
    """Helper function to loop over all functions and plot them, along with the error"""

    for name, ref_data in ref.items():
        plot_f_and_errors(ref_data,
                          test[name],
                          error[name],
                          name,
                          observable)


def plot_f_and_errors(ref, test, error, title, observable):
    """Function to handle the plotting of the functions"""

    # Plot settings
    cm2in = 1/2.54
    fig = plt.figure(figsize=(8*cm2in, 12*cm2in), constrained_layout=True)
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2., 1.])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # Plot reference and test property
    ax0.plot(ref[0], ref[1], color='black',
             label="Reference " + str(title), lw=2)
    ax0.plot(test[0], test[1], color='red', dashes=(0.5, 1.5), dash_capstyle='round',
             label="Test " + str(title), lw=2)

    # Plot error
    ax1.plot(error[0], error[1], color='black', lw=2)

    # Formatting
    ax0.set_ylabel(observable)
    ax0.set_xticklabels([])
    ax1.set_ylabel("Absolute Error")
    if observable == 'VDOS':
        ax0.set_ylim(ymin=0)
        ax0.set_xlim([0,4500])
        ax1.set_xlim([0,4500])
        ax1.set_xlabel(r"Frequency (cm$^{-1}$)")
    if observable == 'RDF':
        ax1.set_xlabel(r'Distance ($\mathrm{\AA{}}$)')

    ax0.set_title("Species: " + str(title))

    plt.savefig(observable+"-" + str(title) + ".pdf")


def plot_mae_errors(error, fn_out='accuracy-all.pdf'):
    """Function to handle the plotting of the summed absolute errors
    for each element."""

    def autolabel(rects, form=r'{:1.1f}'):
        """Attach a text label in each bar in *rects*, displaying its value."""
        for rect in rects:
            width = rect.get_width()
            plt.annotate(form.format(width),
                         xy=(0.0,rect.get_y() + rect.get_height() / 2),
                         xytext=(3, 0),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='left', va='center',
                         color='w', fontsize=8, fontweight='bold')

    # Plot settings
    cm2in = 1/2.54
    cm = plt.cm.viridis(np.linspace(0,1.0,8))[::-1]
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True,
                           figsize=(8*cm2in, 6*cm2in))

    # height of the bars
    height = 0.6
    # Label and their locations
    y = np.arange(len(error) + 1)
    labels = [key for key in error.keys()]

    # Convert errors to percent
    errors = np.array([(1 - error[key][2]) * 100 for key in labels][::-1])

    # Plot individual errors and mean
    rects = ax.barh(y[1:], errors, height, color=cm[4])
    autolabel(rects)
    rects = ax.barh(y[0], errors.mean(), height, color=cm[6])
    autolabel(rects)

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(np.append(labels,'All')[::-1])
    ax.set_xlim([0,100])
    ax.set_xlabel("Accuracy (%)")
    ax.set_frame_on(False)
    ax.grid(axis='x')

    if fn_out is not None:
        plt.savefig(fn_out)
