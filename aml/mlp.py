"""Handle different machine learning potentials."""

__all__ = [
    'N2P2'
]

import shutil
from pathlib import Path
from subprocess import Popen

from .acsf import format_combine_ACSFs, generate_radial_angular_default
from .data import atomic_numbers
from .io import from_file, to_file
from .utilities import prepare_command_mpi


class MLP:
    """Base class for representations of machine learning potentials.

    Functionality shared by all MLPs lives here.
    """

    def __init__(self, elements, n):

        # list of elements included in the model
        # (Ondrej would really like this to be kinds, rather than elements,
        # but that does not seem to be supported by programs at the moment.)
        self._elements = tuple(elements)

        # number of committee members
        self._n = n

    @property
    def elements(self):
        return self._elements

    @property
    def n(self):
        return self._n

    @property
    def is_trained(self):
        raise NotImplementedError


class MLPProcess(MLP):
    """Machine learning potential ran by launching a separate process.

    This should have all the functionality related to launching processes, file I/O and such.
    """

    @classmethod
    def from_directories(cls, directories, *args, **kwargs):
        """Read model parameters from existing directories.

        The first argument is the list of directories to read from. The remaining
        arguments are all passed to the constructor of the model. The number of directories
        must match the size of the committee.
        """

        # construct the model
        model = cls(*args, **kwargs)

        # check consistency
        ln = len(directories)
        n = model.n
        if ln != n:
            raise ValueError(
                'Number of directories ({l:d}) does not match the size of the committee ({n:d}).')

        # make sure the directories are Path objects
        directories = [Path(d) for d in directories]

        # read in the parameters
        model._read_parameters(directories)

        return model

    def __init__(self, elements, n, dir_run='.', n_tasks=1, remove_output=False):
        """Process-based machine learning potential.

        Arguments:
            elements:
            n:
            dir_run:
            n_tasks:
            remove_output:
        """

        super().__init__(elements, n)

        # run everything relative to this directory
        self.dir_run = Path(dir_run)

        # keywords set in specific MLPs
        self.model_parameters = None

        # number of batches and number of tasks in each batch
        n_tasks = int(n_tasks)
        self.n_tasks = n_tasks
        if n % n_tasks != 0:
            raise ValueError('Number of simultaneous tasks does not divide the number of committee members.')
        self.n_batch = n // n_tasks

        self.remove_output = remove_output

        # stuff to have here:
        # - command (including possible MPI)
        # - names and format of required files and arguments for training and prediction

    @property
    def is_trained(self):
        return self.model_parameters is not None

    def _write_input_train(self, i, directory, structures, n_epoch, label_prop):
        """Write input files for a single training run.

        Needs to be implemented for a specific program.
        """

        raise NotImplementedError

    def _read_output_train(self, directory):
        """Read output files for a single training run.

        Needs to be implemented for a specific program.
        """

        raise NotImplementedError

    def _write_input_predict(self):
        """Write input files for a single prediction run.

        Needs to be implemented for a specific program.
        """

        raise NotImplementedError

    def _read_output_predict(self):
        """Read output files for a single prediction run.

        Needs to be implemented for a specific program.
        """

        raise NotImplementedError

    def _launch(self, operation, i_task, directory):
        """Launch a single run of the MLP.

        Needs to be implemented for a specific program.
        """

        raise NotImplementedError

    def _read_parameters(self, directories):
        """Read contents of files with model parameters."""

        model_parameters = []
        for directory in directories:
            model_parameters.append(self._read_output_train(directory))
        self.model_parameters = model_parameters

    def train(self, structures, n_epoch, label_prop='reference'):
        """Perform optimization of n MLPs."""

        # prepare settings specific to training
        operation = 'train'
        fmt_dir = 'train-{:03d}'

        # check here that we're not trained yet
        # (Maybe allow explicit forced override?)
        if self.is_trained:
            raise Exception("Model already trained.")

        # loop over all members to prepare run directories
        directories = []
        for i in range(self.n):
            # prepare the run directory itself
            directory = self.dir_run / fmt_dir.format(i)
            directory.mkdir(parents=True, exist_ok=True)
            directories.append(directory)
            # prepare input files in the run directory
            self._write_input_train(i, directory, structures, n_epoch, label_prop)

        # prepare iterator over directories
        i_directories = iter(directories)

        # loop over members in a structured way - batches of tasks to run
        for i_batch in range(self.n_batch):

            # launch all tasks in this batch
            tasks = []
            for i_task in range(self.n_tasks):
                tasks.append(self._launch(operation, i_task, next(i_directories)))

            # wait for all tasks in this batch to finish
            # Check for errors - for now, just mention this to the user, but we will have to do better.
            # Also, this is useless if the call is several shell commands together.
            for task in tasks:
                task.wait()
                if task.returncode != 0:
                    print(f'Command finished with a non-zero return code: {task.returncode:d}')
                    print(task)

        # read resulting model parameters
        self._read_parameters(directories)

    def predict(self, structures, label='predict'):
        """Perform predictions for n MLPs."""

        operation = 'predict'

        # can only predict if the model was trained before
        if not self.is_trained:
            raise Exception('Model is not trained, can not predict.')

        # loop over all members to prepare run directories
        directories = []
        for i in range(self.n):
            # prepare the run directory itself
            directory = self.dir_run / f'{label:}-{i:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            directories.append(directory)
            # prepare all input files in the run directory
            self._write_input_predict(i, directory, structures)

        # prepare iterator over directories
        i_directories = iter(directories)

        # loop over members in a structured way - batches of tasks to run
        for i_batch in range(self.n_batch):

            # launch all tasks in this batch
            tasks = []
            for i_task in range(self.n_tasks):
                tasks.append(self._launch(operation, i_task, next(i_directories)))

            # wait for all tasks in this batch to finish
            # Check for errors - for now, just mention this to the user, but we will have to do better.
            # Also, this is useless if the call is several shell commands together.
            for task in tasks:
                task.wait()
                if task.returncode != 0:
                    print(f'Command finished with a non-zero return code: {task.returncode:d}')
                    print(task)

        # update structures with the just performed prediction
        for i in range(self.n):
            self._read_output_predict(directories[i], structures)

        # optionally remove all directories
        if self.remove_output:
            for directory in directories:
                shutil.rmtree(directory)

    def save_model(self, label='model'):
        """Save model to use in external software."""

        # can only save if the model was trained before
        if not self.is_trained:
            raise Exception('Model is not trained, can not save.')

        # loop over all members to save
        directories = []
        for i in range(self.n):
            # prepare the directory itself
            directory = self.dir_run / label / f'nnp-{i:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            directories.append(directory)
            # prepare all input files in the run directory
            self._write_input_predict(i, directory, structures=None)


class N2P2(MLPProcess):
    """Representation of the N2P2 program."""

    # name of input file to write
    fn_input = 'input.nn'

    # name of data file to write
    fn_input_data = 'input.data'

    # name of file with scaling parameters
    fn_scaling = 'scaling.data'

    # templates for names of files with weights and biases
    fn_w_template = 'weights.{Z:03d}.data'
    fn_w_glob = 'weights.{Z:03d}.*.out'

    # name of file with predicted energies
    energy_file_name = 'energy.comp'

    # name of file with predicted forces
    forces_file_name = 'forces.comp'

    # formats of files
    format_structures = 'RuNNer'
    format_energies = 'N2P2_E'
    format_forces = 'N2P2_F'

    # command templates
    template_cmd = {
        'train': (
            "{cmd_mpi:s} nnp-scaling 100 > nnp-scaling-stdout.log 2> nnp-scaling-stdout.err; "
            "{cmd_mpi:s} nnp-train > nnp-train-stdout.log 2> nnp-train-stdout.err"),
        'predict': '{cmd_mpi:s} nnp-dataset 0 > nnp-dataset-stdout.log 2> nnp-dataset-stdout.err'
    }

    def __init__(
        self,
        elements,
        n,
        fn_template,
        exclude_pairs=None,
        exclude_triples=None,
        dir_run='.',
        n_tasks=1,
        n_core_task=1,
        node_size=None,
        remove_output=False
    ):
        """Initialize this MLP object.

        A total of `n` tasks need to be performed for each operation, `n_tasks` determines how many
        can run simultaneously. `n` should be divisible by `n_tasks`.

        This will launch n2p2 in parallel using MPI if `n_core_task>1` and potentially on multiple nodes
        if `node_size` is set. In that case, OpenMPI rankfiles are used to set up the runs.

        Arguments:
            elements: list of elements described by the model
            n: number of committee members
            fn_template: name of file with template of n2p2 input
            exclude_pairs: a list of element pairs to not consider (optional)
            exclude_triples: a list of element triples to not consider (optional)
            dir_run: directory to run everything in
            n_tasks: number of tasks to run simultaneously
            n_core_task: number of cores per task
            node_size: number of cores on a node, if we are to run on multiple nodes
            remove_output: whether to remove training and prediction run directories
        """

        super().__init__(elements, n, dir_run, n_tasks, remove_output)

        # save list of pairs and triples to exclude from ACSF generation
        self.exclude_pairs = exclude_pairs
        self.exclude_triples = exclude_triples

        # read input file template from a file
        self.template = from_file(fn_template)

        # This is passed to the template for input.nn, but not exposed to the user
        self.keywords = {}

        # determine parallelization mode and store related settings
        if node_size is None:
            # node size is not specified, we're running on a single node only
            n_core_task = int(n_core_task)
            if n_core_task == 1:
                # each task will be serial, multiple can still run simultaneously
                mode = 'serial'
            else:
                # each task will be MPI parallel
                mode = 'OpenMPI-single'
        else:
            # we have a node size, prepare to run on multiple nodes
            # MPI command expects a rankfile in the working directory
            node_size = int(node_size)
            mode = 'OpenMPI-multi'
        self.mode = mode
        self.node_size = node_size
        self.n_core_task = n_core_task

    def _write_input_nn(self, fn, seed=0, n_epoch=1):
        """Write the main n2p2 input file `input.nn`."""

        elements = self.elements
        str_elements = ' '.join(elements)
        n_elements = len(elements)
        exclude_pairs = self.exclude_pairs
        exclude_triples = self.exclude_triples

        # generate default ACSFs - will be used if the `acsf` formatting field is present
        radials, angulars = generate_radial_angular_default()
        acsf = format_combine_ACSFs(radials, angulars, elements, exclude_pairs, exclude_triples)

        # prepare input files in the run directory
        str_input = self.template.format(
            n_elements=n_elements,
            elements=str_elements,
            seed=seed,
            n_epoch=n_epoch,
            acsf=acsf,
            **self.keywords)
        to_file(str_input, fn)

    def _write_input_train(self, i, directory, structures, n_epoch, label_prop):
        """Write input files for a single n2p2 training run."""

        # prepare and write input file
        self._write_input_nn(directory / self.fn_input, seed=i, n_epoch=n_epoch)

        # write structures file
        structures.to_file(directory / self.fn_input_data, fformat=self.format_structures, label_prop=label_prop)

    def _read_output_train(self, directory):
        """Read output files for a single n2p2 training run."""

        # Here we find the highest available epoch.
        # In principle, we could read all epochs, though.

        parameters = {}

        # load scaling data
        parameters[self.fn_scaling] = from_file(directory / self.fn_scaling, binary=True)

        # load weights data
        epochs = []
        for element in self.elements:

            # nothing found yet
            fn = None

            # atomic number for this element name
            Z = atomic_numbers[element]

            # weights file name for prediction
            fn_w_predict = self.fn_w_template.format(Z=Z)

            # first, look for highest epoch-numbered file that exists
            fns_w = sorted(directory.glob(self.fn_w_glob.format(Z=Z)))
            if len(fns_w) > 0:
                # found it!
                fn = fns_w[-1]
                # extract the epoch number from it
                epoch = int(fn.name.split('.')[-2])
                epochs.append(epoch)

            # if none exist, look for file name for prediction
            if fn is None:
                fns_w = list(directory.glob(fn_w_predict))
                if len(fns_w) > 0:
                    fn = fns_w[-1]
                    # epoch unknown, indicate that
                    epochs.append(None)

            # if we still have nothing, it's a failure
            if fn is None:
                raise Exception(f'No weights file found for element: {element:s}')

            # store the file under the name needed for prediction
            parameters[fn_w_predict] = from_file(fn, binary=True)

        return parameters

    def _write_input_predict(self, i, directory, structures):
        """Write input files for a single n2p2 prediction run."""

        # prepare and write input file
        self._write_input_nn(directory / self.fn_input)

        # write structures file
        if structures is not None:
            structures.to_file(directory / self.fn_input_data, fformat=self.format_structures)

        # write parameters files
        for fn, parameters in self.model_parameters[i].items():
            to_file(parameters, directory / fn, binary=True)

    def _read_output_predict(self, directory, structures):
        """Read output files for a single n2p2 prediction run."""

        label = directory.name

        # read energies, store them with the original structures
        structures.update_from_file(
            directory / self.energy_file_name, fformat=self.format_energies, label_prop=label, force=True)

        # read forces, store them with the original structures
        structures.update_from_file(
            directory / self.forces_file_name, fformat=self.format_forces, label_prop=label, force=True)

    def _launch(self, operation, i_task, directory):
        """Launch a single n2p2 run.

        Arguments:
            operation:
            i_task:
            directory:
        """

        # prepare MPI launch command
        details = False
        cmd_mpi = prepare_command_mpi(
            i_task,
            n_core_task=self.n_core_task,
            node_size=self.node_size,
            mode=self.mode,
            details=details,
            fn_rank=directory / 'rankfile.txt'
        )

        # launch the task
        cmd = self.template_cmd[operation].format(cmd_mpi=cmd_mpi).strip()
        task = Popen(cmd, cwd=directory, shell=True)

        return task
