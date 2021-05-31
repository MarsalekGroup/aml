"""Various utilities needed elsewhere in the code."""

__all__ = [
    'AMLError', 'AMLInternalError', 'AMLTrainingError', 'AMLPredictionError',
    'view_read_only',
    'write_rankfile_openmpi', 'prepare_command_mpi'
]


class AMLError(Exception):
    """Any AML-related error."""
    pass


class AMLInternalError(AMLError):
    """An error in the code of AML itself."""
    pass


class AMLIOError(AMLError):
    """An error in I/O operations."""
    pass


class AMLTrainingError(AMLError):
    """Problem with training a model in AML."""
    pass


class AMLPredictionError(AMLError):
    """Problem with performing a model prediction in AML."""
    pass


def view_read_only(data):
    """Return a read-only view of the array `data`."""
    if data is None:
        return None
    else:
        data = data.view()
        data.flags.writeable = False
        return data


def write_rankfile_openmpi(fn, job_index: int, job_size: int, node_size: int):
    """Write one rankfile in the OpenMPI format.

    In this context, one "job" is one MPI run of `job_size` processes. Based on the size of one compute node,
    further jobs are placed on nodes with higher numbers (in the OpenMPI relative numbering scheme).

    Arguments:
        fn: name of output file
        job_index: index of the job, starts at 0
        job_size: number of processes in the job
        node_size: size of one compute node
    """

    # format of one line, as per OpenMPI docs
    fmt_line = 'rank {i_rank:d}=+n{i_node:d} slot={i_core:d}\n'

    # starting node and core
    i_core_global = job_index * job_size
    i_node = i_core_global // node_size
    i_core = i_core_global % node_size

    # one line per process in job
    with open(fn, 'w') as f_out:
        for i_rank in range(job_size):
            f_out.write(fmt_line.format(i_rank=i_rank, i_node=i_node, i_core=i_core))
            i_core += 1
            if i_core >= node_size:
                i_core = 0
                i_node += 1


def prepare_command_mpi(
    i_task: int,
    n_core_task: int = 1,
    node_size: int = 1,
    mode: str = 'serial',
    details: bool = False,
    fn_rank=None
):
    """Prepare a command to launch a process in parallel using MPI.

    Given the index of the task, the number of cores to use for each task, and the size of one node,
    prepare the command to lauch a task in parallel using MPI. In the case of multi-node OpenMPI,
    a rank file also needs to be written to disk - `fa_rank` needs to be specified for that.

    The available modes are:
        serial: no MPI, empty string
        MPI: plain `mpirun --np <n>`, no binding
        OpenMPI-single: OpenMPI, binding suitable for a single node
        OpenMPI-multi: OpenMPI, binding suitable for multiple nodes using rank files
        Slurm: launch and bind using Slurm's `srun`

    Arguments:
        i_task: index of this task
        n_core_task: number of cores per task
        node_size: number of cores on each node
        mode: the specific MPI launch method to use
        details: whether to request binding details to be reported
        fn_rank: name of OpenMPI rank file to write to disk

    Returns:
        MPI launch command as a string.
    """

    if mode == 'serial':
        if n_core_task != 1:
            raise ValueError('Impossible to use more than one core per task in a serial run.')
        cmd_mpi = ''

    elif mode == 'MPI':
        cmd_mpi = f'mpirun --np {n_core_task:d}'

    elif mode == 'OpenMPI-single':
        if node_size is not None:
            if i_task * n_core_task > node_size:
                msg = f'{node_size:d} cores on one node not enough for task {i_task:d} on {n_core_task:d} cores.'
                raise ValueError(msg)
        cpu_set = f'{i_task*n_core_task}-{(i_task+1)*n_core_task-1}'
        cmd_mpi = f'mpirun --np {n_core_task:d} --bind-to cpulist:ordered --map-by hwthread --cpu-set {cpu_set:s}'

    elif mode == 'OpenMPI-multi':

        # prepare OpenMPI rankfile
        if fn_rank is None:
            raise ValueError('Mode "OpenMPI-multi" requires `fn_rank` to be specified.')
        write_rankfile_openmpi(fn=fn_rank, job_index=i_task, job_size=n_core_task, node_size=node_size)

        # MPI command itself
        cmd_mpi = f'mpirun --np {n_core_task:d} --bind-to core --rankfile rankfile.txt'

    elif mode == 'Slurm':
        raise NotImplementedError('Slurm support not implemented.')

    else:
        raise ValueError(f'Unrecognized mode: {mode:s}')

    # detailed information on MPI binding
    if details:
        if mode.startswith('OpenMPI'):
            cmd_mpi += ' --display-devel-map --display-allocation --report-bindings'
        else:
            raise ValueError(f'No details available for mode: {mode:s}')

    return cmd_mpi
