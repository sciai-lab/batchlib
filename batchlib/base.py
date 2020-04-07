import os
import json

from abc import ABC
from glob import glob

from .util import open_file


class BatchJob(ABC):
    """ Batch job base class.

    Batch jobs process all files in an input directory. A batch job can be
    called repeatedly on the same directory and will only process the files for which
    output has not been computed yet.

    Deriving classes must have the member `runners` dict[str, function].
    This dictionary maps computation target (e.g. local execution, slurm cluster) to
    function executing the job for this target.
    Minimal implementation:
        self.runners = {'default': self.run}
    The run functions must have the syntax:
        def run(self, input_files, output_files, **kwargs)

    Deriving classes may override the following methods.
    - check output    - check if output is present
    - validate_input  - check that input is valid
    - validate_output - check that output is valid
    Important: validate_input and check_output should be fast

    Status layout:
    {'state': <'continue', 'invalid_inputs', 'failed_outputs'>,
     'invalid_inputs': [],  # list of inputs that are not available or have an issue
     'failed_outputs': []}  # list of outputs that were not computed or have an issue
    """

    @staticmethod
    def check_keys(keys):
        if keys is None:
            return None, None

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

        if isinstance(ndim, list):
            if len(ndim) != len(keys):
                raise ValueError("Ivnalid data ndim")
            exp_ndim = ndim
        else:
            exp_ndim = [ndim] * len(keys)
        return ndim, exp_ndim

    # TODO consider changing default file format to n5
    def __init__(self, input_key=None, output_key=None,
                 input_ndim=None, output_ndim=None,
                 input_pattern='*.h5', output_ext=None,
                 target='default'):
        # the input and output keys (= internal datasets)
        # the _exp_ variables are the normalized versions we need in the checks
        self.input_key, self._input_exp_key = self.check_keys(input_key)
        self.output_key, self._output_exp_key = self.check_keys(output_key)

        # the input and output dimensions
        # the _exp_ variables are the normalized versions we need in the checks
        self.input_ndim, self._input_exp_ndim = self.check_ndim(input_ndim, self._input_exp_key)
        self.output_ndim, self._output_exp_ndim = self.check_ndim(output_ndim, self._output_exp_key)

        self.input_pattern = input_pattern
        self.input_ext = os.path.splitext(self.input_pattern)[1]
        self.output_ext = self.input_ext if output_ext is None else output_ext
        self.target = target

    @property
    def name(self):
        name_ = self.__class__.__name__
        # if the class has an identifier member, we add it to the name
        # this allows running multiple batch jobs of the same type for one
        # experiment, by adding the identifiers
        if hasattr(self, 'identifier'):
            identifier = self.identifier
            assert isinstance(identifier, str)
            return name_ + identifier
        else:
            return name_

    def status_file(self, folder):
        return os.path.join(folder, self.name + '.status')

    def get_status(self, folder):
        stat_file = self.status_file(folder)
        if os.path.exists(stat_file):
            with open(stat_file) as f:
                status = json.load(f)
        else:
            status = {}
        return status

    def update_status(self, folder, status,
                      invalid_inputs=None, failed_outputs=None, continue_=None):
        # TODO check that only one of the three last inputs is not None
        path = self.status_file(folder)

        if invalid_inputs is not None:
            status['state'] = 'invalid_inputs'
            status['invalid_inputs'] = invalid_inputs

        if failed_outputs is not None:
            status['state'] = 'failed_outputs'
            status['failed_outputs'] = failed_outputs

        if continue_ is not None:
            status['state'] = 'continue'

        with open(path, 'w') as f:
            json.dump(status, f, indent=2, sort_keys=True)

    def to_inputs(self, outputs, input_folder):
        names = [os.path.splitext(os.path.split(out)[1])[0] for out in outputs]
        inputs = [os.path.join(input_folder, name + self.input_ext) for name in names]
        return inputs

    def to_outputs(self, inputs, folder):
        names = [os.path.splitext(os.path.split(inp)[1])[0] for inp in inputs]
        outputs = [os.path.join(folder, name + self.output_ext) for name in names]
        return outputs

    def get_inputs(self, folder, input_folder, status, force_recompute):
        state = status.get('state', 'continue')

        in_pattern = os.path.join(input_folder, self.input_pattern)
        input_files = glob(in_pattern)

        # check if we have invalid inputs
        invalid_inputs = self.get_invalid_inputs(input_files)
        if len(invalid_inputs) > 0:
            if state == 'invalid_inputs':
                prev_invalid = len(status['invalid_inputs'])
                msg = "%i inputs are invalid from %i in previous call" % (len(invalid_inputs), prev_invalid)
            else:
                msg = "%i inputs are invalid, fix them and rerun this task" % len(invalid_inputs)
            self.update_status(folder, status, invalid_inputs=invalid_inputs)
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
            failed_outputs = state['failed_outputs']
            additional_inputs = set(self.to_inputs(failed_outputs, input_folder))
            input_files = list(set(input_files).union(additional_inputs))

        return input_files

    def __call__(self, folder, input_folder=None, force_recompute=False, **kwargs):
        os.makedirs(folder, exist_ok=True)
        status = self.get_status(folder)

        # the actual input folder we use
        input_folder_ = folder if input_folder is None else input_folder

        # validate and get the input files to be processed
        input_files = self.get_inputs(folder, input_folder_, status, force_recompute)
        if len(input_files) == 0:
            return

        output_files = self.to_outputs(input_files, folder)

        # get the function to run the actual job
        # runners is a dict mapping the computation target (e.g. 'default', 'slurm')
        # to the correct run  function.
        # if the target is not available, it defaults to the default run implementation,
        # but throws a warning
        _run = self.runners.get(self.target, None)
        if _run is None:
            raise RuntimeError("%s does not implement a runner for %s" % (self.name, self.target))

        # TODO better exception handling
        try:
            _run(input_files, output_files, **kwargs)
        except Exception:
            pass

        # TODO output validation can be expensive, so we might want to parallelize
        # validate the outputs and update the status
        failed_outputs = self.get_invalid_outputs(output_files)
        if len(failed_outputs) > 0:
            state = status.get('state', 'continue')
            if state == 'failed_outputs':
                prev_failed = len(status['failed_outputs'])
                msg = "%i outpus have failed from %i in previous call" % (len(failed_outputs), prev_failed)
            else:
                msg = "%i outputs have failed" % len(failed_outputs)
            self.update_status(folder, status, failed_outputs=failed_outputs)
            raise RuntimeError(msg)

        # if everything went through, we set the state to 'continue'
        # which means we accept more data
        self.update_status(folder, status, continue_=True)

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
