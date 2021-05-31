"""Query by committee."""

__all__ = [
    'QbC'
]

import shelve
import time
from pathlib import Path

import numpy as np
import scipy.stats

from . import constants
from .structures import Structures


def moment_std(data, n=6):
    return scipy.stats.moment(np.std(data, axis=0), n, axis=None)


def mean_std(data):
    return np.mean(np.std(data, axis=0))


def evaluate_cda_forces(structures, label_prop):
    """Convenience function to evaluate disagreement and convert units."""
    cda = structures.reduce_property(f_reduce=mean_std, name_prop='forces', label_prop=label_prop).mean()
    cda *= 1000 * constants.angstrom / constants.eV
    return cda


class QbC:
    """Query by committee for NNP training sets."""

    def __init__(
        self,
        structures, cls_model, kwargs_model,
        n_add, n_epoch, n_iterations, n_candidate,
        n_train_initial=0, train_set_initial=None, fn_results='results.shelf',
        fn_restart=None
    ):
        """Store all settings for the QbC process."""

        # store everything
        self._structures = structures
        self._cls_model = cls_model
        self._kwargs_model = kwargs_model
        self._n_add = n_add
        self._n_epoch = n_epoch
        self._n_iterations = n_iterations
        self._n_candidate = n_candidate
        self._n_train_initial = n_train_initial
        self._fn_results = fn_results
        self._fn_restart = fn_restart

        # store initial training set - provided or empty
        if train_set_initial is None:
            train_set_initial = Structures([])
        self._train_set_initial = train_set_initial

        print('Query by Committee')
        print('==================')
        print()
        print('Selecting from a data set with existing reference values.')
        print(f'Committee size: {kwargs_model["n"]:d}')
        print(f'Training structures to select each iteration: {n_add:d}')
        print(f'Structures to randomly sample as candidates: {n_candidate:d}')
        print(f'Number of QbC iterations: {n_iterations:d}')
        print()

    def run(self):

        # get back settings - this will need updating
        structures = self._structures
        cls_model = self._cls_model
        kwargs_model = self._kwargs_model
        n_add = self._n_add
        n_epoch = self._n_epoch
        n_iterations = self._n_iterations
        n_candidate = self._n_candidate
        n_train_initial = self._n_train_initial
        fn_results = self._fn_results
        train_set_initial = self._train_set_initial
        fn_restart = self._fn_restart

        # iteration and training set to start from
        if fn_restart is None:
            i_iteration_init = 1
            print(f'Initial training set of {len(train_set_initial):d} structures loaded.')
            idx_train = structures.sample(n_train_initial)
            print(f'Additionally, {n_train_initial:d} structures were sampled randomly.')
        else:
            print('Reading restart information...')
            with shelve.open(fn_restart) as shelf:
                label_last = sorted(shelf.keys())[-1]
                last = shelf[label_last]
            i_iteration_init = last['i_iteration'] + 1
            idx_train = last['idx_train']
            print(f'  Starting at iteration {i_iteration_init:d}.')
            print(f'  Read {len(idx_train):d} training structures.')
        print()

        # initial training structures
        structures_train = train_set_initial + structures.get(idx_train)
        if len(structures_train) == 0:
            raise ValueError(
                'Size of initial training set is zero. '
                'Provide either initial training structures or specify random sampling'
            )

        # The selection of new structures at the end of the last iteration is not used for anything,
        # but it is still useful to know the prediction and thus disagreement even for that last iteration.

        print('QbC main loop')
        print('-------------')
        print()

        # main active learning loop
        for i_iteration in range(i_iteration_init, n_iterations+1):

            time0 = time.time()

            print(f'QbC iteration {i_iteration:}')
            print(f'Training set size: {len(idx_train)}')

            # sample new structures as candidates for selection
            idx_candidate = structures.sample(n_candidate)
            structures_candidate = structures.get(idx_candidate)

            # a new directory to run this iteration
            dir_run = Path(f'iteration-{i_iteration:03d}')

            # construct a new committee model
            model = cls_model(dir_run=dir_run, **kwargs_model)

            # run training
            print('Training committee... ', end='', flush=True)
            model.train(structures_train, n_epoch)
            print('done.')

            # run prediction for the training set and for candidates
            print('Prediction for training set... ', end='', flush=True)
            model.predict(structures_train, label='eval_train')
            print('done.')
            print('Prediction for candidate set... ', end='', flush=True)
            model.predict(structures_candidate)
            print('done.')

            # calculate mean committee disagreement
            cda = evaluate_cda_forces(structures_candidate, 'pred*')
            cda_train = evaluate_cda_forces(structures_train, 'eval_train*')

            # select highest disagreement structures, calculate their disagreement
            idx_selected = structures_candidate.select_highest_error(
                n=n_add, label_prop='pred*', f_reduce=moment_std, name_prop='forces')
            # map these indices to original structure indices
            idx_new = [idx_candidate[i] for i in idx_selected]
            cda_new = evaluate_cda_forces(structures.get(idx_new), 'pred*')

            # combine with previous training set, do not add again structures that are already present
            idx_unique = list(set(idx_new) - set(idx_train))
            idx_train += idx_unique
            structures_train = structures.get(idx_train)
            structures_train += train_set_initial
            print(f'Selected {len(idx_new):d} structures, {n_add-len(idx_unique):d} already present in training set.')
            print(f'Adding {len(idx_unique):d} new structures to training set.')

            # compose and save results for this iteration
            # useful for restarts and postprocessing
            results = {
                'i_iteration': i_iteration,
                'idx_train': idx_train,
            }
            label = f'iteration-{i_iteration:03d}'
            with shelve.open(fn_results) as shelf:
                shelf[label] = results

            # report some info
            print("Mean standard deviation of force committee disagreement [meV/A]")
            print("   train       candidates  selected")
            print(f"   {cda_train:9.3e}   {cda:9.3e}   {cda_new:9.3e}")

            # we are done with this iteration
            time_total = time.time() - time0
            print(f'QbC iteration finished in {time_total:.0f} s.\n')
