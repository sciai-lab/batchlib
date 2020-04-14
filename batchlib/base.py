import os
import json

from abc import ABC
from glob import glob

from .util import open_file, get_file_lock, write_viewer_settings


class BatchJob(ABC):
    """ Batch job base class.

    Batch jobs process all files in an input directory. A batch job can be
    called repeatedly on the same directory and will only process the files for which
    output has not been computed yet.
    Advanced execution modes (enabled by passing the corresponding flags to __call__)
    - force_recompute - recompute all outputs, even if they are present already
    - ignore_invalid_inputs - run computation for the valid inputs if there are invalid ones
                              (in default mode the job will raise a RuntimeError in this case)
    - ignore_failed_outputs - continue running even if some outputs were not computed properly
                              (in default mode the job will raise a RuntimeError in this case)

    Deriving classes must implement a run method, that implements the actual computation.
    It must follow the syntax
        def run(self, input_files, output_files, **kwargs)
    where input_files is the list of files to process
    and output_files the files to write the correspondign results.

    Deriving classes may override the following methods.
    - check output    - check if output is present
    - validate_input  - check that input is valid
    - validate_output - check that output is valid
    Important: validate_input and check_output should be fast

    Status layout:
    {'state': <'processed', 'errored', 'invalid_inputs', 'failed_outputs'>,
     'invalid_inputs': [],  # list of inputs that are not available or have an issue
     'failed_outputs': []}  # list of outputs that were not computed or have an issue
    """
    # by default, we lock the whole folder and don't need to lock the individual jobs
    lock_job = False

    def __init__(self, input_pattern, output_ext=None, identifier=None):
        if not hasattr(self, 'run'):
            raise AttributeError("Class deriving from BatchJob must implement run method")

        self.input_pattern = input_pattern
        self.input_ext = os.path.splitext(self.input_pattern)[1]
        self.output_ext = self.input_ext if output_ext is None else output_ext

        is_none = identifier is None
        is_str = isinstance(identifier, str)
        if not (is_none or is_str):
            raise ValueError("Expect identifier to  be None or string, not %s" % type(identifier))
        self.identifier = identifier

    @property
    def name(self):
        name_ = self.__class__.__name__
        # if the class identifier is not None, it's added to the name
        # this allows running multiple batch jobs of the same type for one class
        return name_ if self.identifier is None else name_ + self.identifier

    def status_file(self, folder):
        return os.path.join(folder, 'batchlib', self.name + '.status')

    def lock(self, folder):
        lock_path = os.path.join(folder, 'batchlib', self.name + '.lock')
        return get_file_lock(lock_path, self.lock_job)

    def get_status(self, folder):
        stat_file = self.status_file(folder)
        if os.path.exists(stat_file):
            with open(stat_file) as f:
                status = json.load(f)
        else:
            status = {}
        return status

    def update_status(self, folder, status,
                      invalid_inputs=None, failed_outputs=None, processed=None):
        # TODO check that only one of the three last inputs is not None
        path = self.status_file(folder)

        if invalid_inputs is not None:
            status['state'] = 'invalid_inputs'
            status['invalid_inputs'] = invalid_inputs

        if failed_outputs is not None:
            status['state'] = 'failed_outputs'
            status['failed_outputs'] = failed_outputs

        if processed is not None:
            status['state'] = 'processed'

        with open(path, 'w') as f:
            json.dump(status, f, indent=2, sort_keys=True)
        return status

    def to_inputs(self, outputs, input_folder):
        names = [os.path.splitext(os.path.split(out)[1])[0] for out in outputs]
        inputs = [os.path.join(input_folder, name + self.input_ext) for name in names]
        return inputs

    def to_outputs(self, inputs, folder):
        names = [os.path.splitext(os.path.split(inp)[1])[0] for inp in inputs]
        outputs = [os.path.join(folder, name + self.output_ext) for name in names]
        return outputs

    def get_inputs(self, folder, input_folder, status, force_recompute, ignore_invalid_inputs):
        state = status.get('state', 'processed')

        in_pattern = os.path.join(input_folder, self.input_pattern)
        input_files = glob(in_pattern)
        # check if we have invalid inputs
        invalid_inputs = self.get_invalid_inputs(input_files)
        if len(invalid_inputs) > 0:
            if state == 'invalid_inputs':
                prev_invalid = len(status['invalid_inputs'])
                msg = "%i inputs are invalid from %i in previous call" % (len(invalid_inputs),
                                                                          prev_invalid)
            else:
                msg = "%i inputs are invalid, fix them and rerun this task" % len(invalid_inputs)
            self.update_status(folder, status, invalid_inputs=invalid_inputs)

            if ignore_invalid_inputs:
                # TODO log warning
                pass
            else:
                raise RuntimeError(msg)

        # force recompute means we just recompute for everything without
        # checking if results are present
        if force_recompute:
            return input_files

        # get the output files corresponding to the inputs and filter for
        # otuputs that are NOT present yet
        output_files = self.to_outputs(input_files, folder)
        output_files = [path for path in output_files if not self.check_output(path)]

        # go back to the inputs corresponding to these output files
        input_files = self.to_inputs(output_files, input_folder)

        # we had failed outputs, we also need to rerun those
        # NOTE that check_output might pass, but validate_output not
        if state == 'failed_outputs':
            failed_outputs = status['failed_outputs']
            additional_inputs = set(self.to_inputs(failed_outputs, input_folder))
            input_files = list(set(input_files).union(additional_inputs))

        return input_files

    def check_outputs(self, output_files, folder, status, ignore_failed_outputs):
        # TODO output validation can be expensive, so we might want to parallelize
        # validate the outputs and update the status
        failed_outputs = self.get_invalid_outputs(output_files)
        if len(failed_outputs) > 0:
            state = status.get('state', 'processed')
            if state == 'failed_outputs':
                n_failed = len(failed_outputs)
                prev_failed = len(status['failed_outputs'])
                msg = "%i outpus have failed from %i in previous call" % (n_failed,
                                                                          prev_failed)
            else:
                msg = "%i outputs have failed" % len(failed_outputs)

            self.update_status(folder, status, failed_outputs=failed_outputs)

            if ignore_failed_outputs:
                # TODO log warning
                pass
            else:
                raise RuntimeError(msg)

    def __call__(self, folder, input_folder=None, force_recompute=False,
                 ignore_invalid_inputs=False, ignore_failed_outputs=False,
                 executable=None, **kwargs):

        # make the work dir, that stores all batchlib status and log files
        work_dir = os.path.join(folder, 'batchlib')
        os.makedirs(work_dir, exist_ok=True)

        # we lock the execution, so that a job with the same name cannot run on
        # this folder at the same time. this allows to start multiple workflows
        # using the same processing step without inteference
        # (this only works properly for n5/zarr files though, because hdf5 doesn't
        # like opening the same file multiple times)
        with self.lock(folder):
            status = self.get_status(folder)

            # the actual input folder we use
            input_folder_ = folder if input_folder is None else input_folder

            # validate and get the input files to be processed
            input_files = self.get_inputs(folder, input_folder_, status,
                                          force_recompute, ignore_invalid_inputs)
            if len(input_files) == 0:
                return status.get('state', 'processed')

            # get the output files corresponding to the input files
            output_files = self.to_outputs(input_files, folder)

            # run the actual computation
            self.run(input_files, output_files, **kwargs)

            # check if all outputs were computed properly
            self.check_outputs(output_files, folder, status, ignore_failed_outputs)

            # if everything went through, we set the state to 'processed'
            status = self.update_status(folder, status, processed=True)
            return status['state']

    def check_output(self, path):
        return os.path.exists(path)

    def validate_input(self, path):
        return os.path.exists(path)

    # in the default implementation, validate_output just calls
    # check_output. This is a separate function though to allow
    # more expensive checks, that are only computed once after
    # the calculation is finished
    def validate_output(self, path):
        return self.check_output(path)

    def get_invalid_inputs(self, inputs):
        return [path for path in inputs if not self.validate_input(path)]

    def get_invalid_outputs(self, outputs):
        return [path for path in outputs if not self.validate_output(path)]


class BatchJobOnContainer(BatchJob, ABC):
    """ Base class for batch jobs operating on at least one
    container file (= h5/n5/zarr file).
    """
    supported_container_extensions = {'.h5', '.hdf5', '.zarr', '.zr', '.n5'}

    def __init__(self, input_pattern, output_ext=None, identifier=None,
                 input_key=None, output_key=None,
                 input_ndim=None, output_ndim=None,
                 viewer_settings={}):
        super().__init__(input_pattern=input_pattern,
                         output_ext=output_ext,
                         identifier=identifier)

        # the input and output keys (= internal datasets)
        # the _exp_ variables are the normalized versions we need in the checks
        self.input_key, self._input_exp_key = self.check_keys(input_key, self.input_ext)
        self.output_key, self._output_exp_key = self.check_keys(output_key, self.output_ext)

        # the input and output dimensions
        # the _exp_ variables are the normalized versions we need in the checks
        self.input_ndim, self._input_exp_ndim = self.check_ndim(input_ndim, self._input_exp_key)
        self.output_ndim, self._output_exp_ndim = self.check_ndim(output_ndim, self._output_exp_key)

        # validate the viewer settings
        for key in viewer_settings:
            if key not in self._output_exp_key:
                raise ValueError("Key %s was not specified in the outputs" % key)
        self.viewer_settings = viewer_settings

    def write_result(self, f, out_key, image, settings=None):
        ds = f.require_dataset(out_key, shape=image.shape, dtype=image.dtype,
                               compression='gzip')
        ds[:] = image
        if settings is None:
            settings = self.viewer_settings.get(out_key, {})
        assert isinstance(settings, dict)
        write_viewer_settings(ds, image, **settings)

    @staticmethod
    def check_keys(keys, ext):
        if keys is None:
            return None, None

        supported_ext = BatchJobOnContainer.supported_container_extensions
        if ext.lower() not in supported_ext:
            ext_str = ", ".join(supported_ext)
            raise ValueError("Invalid container extension %s, expected one of %s" % (ext, ext_str))

        if not isinstance(keys, (str, list)):
            raise ValueError("Invalid data keys")

        if isinstance(keys, list) and not all(isinstance(k, str) for k in keys):
            raise ValueError("Invalid data keys")

        exp_keys = [keys] if isinstance(keys, str) else keys
        return keys, exp_keys

    @staticmethod
    def check_ndim(ndim, keys):
        if ndim is None:
            exp_ndim = None if keys is None else [None] * len(keys)
            return ndim, exp_ndim

        if isinstance(ndim, (list, tuple)):
            if len(ndim) != len(keys):
                raise ValueError("Invalid data ndim")
            exp_ndim = ndim
        else:
            exp_ndim = [ndim] * len(keys)
        return ndim, exp_ndim

    @staticmethod
    def _check_impl(path, exp_keys, exp_ndims):
        if not os.path.exists(path):
            return False

        if exp_keys is None:
            return True

        with open_file(path, 'r') as f:
            for key, ndim in zip(exp_keys, exp_ndims):
                if key not in f:
                    return False
                if ndim is not None and f[key].ndim != ndim:
                    return False
        return True

    def check_output(self, path):
        return self._check_impl(path, self._output_exp_key, self._output_exp_ndim)

    def validate_input(self, path):
        return self._check_impl(path, self._input_exp_key, self._input_exp_ndim)


class BatchJobWithSubfolder(BatchJobOnContainer, ABC):
    """ Base class for a job that operates on an input container
    and writes ouput to a sub-folder.
    """

    def __init__(self, input_pattern, output_ext=None,
                 identifier=None, output_folder="",
                 input_key=None, input_ndim=None):
        self.output_folder = output_folder

        super().__init__(input_pattern, output_ext=output_ext,
                         input_key=input_key, input_ndim=input_ndim)

    def to_outputs(self, inputs, folder):
        names = [os.path.splitext(os.path.split(inp)[1])[0] for inp in inputs]
        outputs = [os.path.join(folder,
                                self.output_folder,
                                name + self.output_ext) for name in names]
        return outputs

    def __call__(self, folder, **kwargs):

        # create output folder
        outdir = os.path.join(folder, self.output_folder)
        os.makedirs(outdir, exist_ok=True)

        super().__call__(folder, **kwargs)
