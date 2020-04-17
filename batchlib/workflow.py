import json
import logging
import os

from batchlib.util.logging import get_logger, add_file_handler, remove_file_handler
from .util import get_file_lock

logger = get_logger('Workflow')


def _dump_status(status_file, status):
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def _update_run_kwargs(run_kwargs, force_recompute,
                       ignore_invalid_inputs, ignore_failed_outputs):
    """ Update the run kwargs with the values for force recompute etc.
    iff the corresponding is not None and not in the kwargs already
    """

    def _update_kwarg(kwarg_name, kwarg):
        if kwarg is not None and kwarg_name not in run_kwargs:
            run_kwargs.update({kwarg_name: kwarg})

    _update_kwarg('force_recompute', force_recompute)
    _update_kwarg('ignore_invalid_inputs', ignore_invalid_inputs)
    _update_kwarg('ignore_failed_outputs', ignore_failed_outputs)

    return run_kwargs


def run_workflow(name, folder, job_dict, input_folder=None, force_recompute=None,
                 ignore_invalid_inputs=None, ignore_failed_outputs=None,
                 lock_folder=True):
    """ Run workflow of consecutive batch jobs.

    The jobs to be run are specified in a dictionary:
    job_dict = {FirstJob: {'build': {...},
                           'run': {...}},
                SecondJob: {}}
    All keys must be classes that inherit from batchlib.BatchJob and map
    to a dictionary with optional keys:
        build_kwargs - keyword arguments passed in class init
        run_kwargs   - keyword arguments passed to call method
        identifier   - extra identifier to make jobs using same class unique

    Arguments:
        name - name of this workflow
        folder - main processing directory
        job_dict - specification of the jobs
        input_folder - separate input folder (default: None)
        force_recompute - whether to recompute all results (default: None)
        ignore_invalid_inputs - whether to continue processing with invalid inputs (default: None)
        ignore_failed_outputs - whether to continue processing with failed outputs (default: None)
        lock_folder - lock the folder so that no other workflow can act on it (default: True)
    """

    work_dir = os.path.join(folder, 'batchlib')
    os.makedirs(work_dir, exist_ok=True)
    # register file workflow's log file
    fh = add_file_handler(logger, work_dir, name)
    # add file handler to tensorboard logger
    logging.getLogger('tensorflow').addHandler(fh)

    lock_file = os.path.join(work_dir, 'batchlib.lock')
    with get_file_lock(lock_file, lock_folder):

        status_file = os.path.join(work_dir, name + '.status')
        status = {}

        logger.info(f"Running workflow: '{name}'. Job spec: {job_dict}")

        for job_id, (job_class, kwarg_dict) in enumerate(job_dict.items()):
            build_kwargs = kwarg_dict.get('build', {})
            run_kwargs = kwarg_dict.get('run', {})

            # if we have a separate input_folder, we assume that it's only passed to
            # the first job, which then needs to take care of
            # copying all files to be processed to the folder
            if job_id == 0 and input_folder is not None:
                run_kwargs.update(dict(input_folder=input_folder))

            job = job_class(**build_kwargs)

            # jobs can be given an identifier, in order to run a job with
            # the same class (but different parameters) twice
            # for example running segmentation.IlastikPrediction
            # for two different projects
            identifier = kwarg_dict.get('identifier', None)
            if identifier is not None:
                job.identifier = identifier
            job_name = job.name

            run_kwargs = _update_run_kwargs(run_kwargs, force_recompute,
                                            ignore_invalid_inputs, ignore_failed_outputs)

            try:
                state = job(folder, **run_kwargs)
            except Exception as e:
                logger.error('Job error', exc_info=True)
                status[job_name] = 'errored'
                _dump_status(status_file, status)
                raise e

            status[job_name] = state
            _dump_status(status_file, status)

        # remove workflow's file handler from the main logger and tf logger
        fh = remove_file_handler(logger, name)
        logging.getLogger('tensorflow').removeHandler(fh)
