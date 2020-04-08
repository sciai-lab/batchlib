import json
import os


def _dump_status(status_file, status):
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


# TODO also accept 'ignore_invalid_inputs' and 'ignore_failed_outputs'
# keywords once this is implemented
def run_workflow(name, folder, job_dict, input_folder=None, force_recompute=None):
    """ Run workflow of consecutive batch jobs.

    The jobs to be run are specified in a dictionary:
    job_dict = {FirstJob: {'build_kwargs': {...},
                           'run_kwargs': {...}},
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
    """

    work_dir = os.path.join(folder, 'batchlib')
    os.makedirs(folder, exist_ok=True)

    status_file = os.path.join(work_dir, name + '.status')
    status = {}

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

        if force_recompute is not None:
            run_kwargs.update(dict(force_recompute=force_recompute))

        try:
            state = job(folder, **run_kwargs)
        except Exception as e:
            status[job_name] = 'errored'
            _dump_status(status_file, status)
            raise e

        status[job_name] = state
        _dump_status(status_file, status)
