import os
import subprocess
from concurrent import futures
from functools import partial

import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file, files_to_jobs, is_group


# TODO
# - ilastik saves files with 'w'. this should be changed to 'a', and then we can write directly
# - how do I run batch processing with n5/zarr files
# - does ilastik support zarr files? (no)
class IlastikPrediction(BatchJobOnContainer):
    """
    """
    def __init__(self, ilastik_bin, ilastik_project,
                 input_key, output_key,
                 keep_channels=None, input_ndim=None, output_ndim=None,
                 **super_kwargs):
        if keep_channels is not None:
            if not isinstance(keep_channels, (list, tuple)):
                raise ValueError("keep_channels must be list or tuple, not %s" % type(keep_channels))
            if not isinstance(output_key, (list, tuple)):
                raise ValueError("If keep_channels is given, output key must be list or tuple")
            n_keep = len(keep_channels)
            if n_keep != len(output_key):
                raise ValueError("Number of output keys and channels to keep does not agree")

        self.keep_channels = keep_channels

        super().__init__(input_key=input_key, output_key=output_key,
                         input_ndim=input_ndim, output_ndim=output_ndim,
                         **super_kwargs)

        self.bin = ilastik_bin
        self.project = ilastik_project

    @staticmethod
    def check_multiscale_input(path, input_key):
        with open_file(path, 'r') as f:
            obj = f[input_key]
        return is_group(obj)

    def predict_images(self, input_files, input_key):
        if len(input_files) == 0:
            return

        is_multi_scale = self.check_multiscale_input(input_files[0], input_key)
        if is_multi_scale:
            inputs = ['%s/%s/s0' % (inp, input_key) for inp in input_files]
        else:
            inputs = ['%s/%s' % (inp, input_key) for inp in input_files]

        cmd = [self.bin, '--headless', '--readonly', '--project=%s' % self.project]
        cmd.extend(inputs)
        subprocess.run(cmd, check=True)

    def save_prediction(self, in_path, out_path, key):
        tmp_path = in_path[:-3] + '-%s_Probabilities.h5' % key
        tmp_key = 'exported_data'

        # load and resave the data
        with open_file(tmp_path, 'r') as f:
            # note: we read from ilastik here, so we don't need `read_image`
            # beacuse this will not be multi-scale
            ds = f[tmp_key]
            data = ds[:]

        if isinstance(self.output_key, str):
            with open_file(out_path, 'a') as f:
                self.write_image(f, self.output_key, data)
        else:
            n_channels = len(data)
            keep_channels = list(range(n_channels)) if self.keep_channels is None else self.keep_channels
            assert all(kc < n_channels for kc in keep_channels)

            with open_file(out_path, 'a') as f:
                current_key_id = 0
                for chan_id, chan in enumerate(data):
                    if chan_id not in keep_channels:
                        continue
                    key = self.output_key[current_key_id]
                    self.write_image(f, key, chan)
                    current_key_id += 1

        # clean up
        os.remove(tmp_path)

    def write_temporary_stack(self, input_files, out_key, n_jobs):

        def _write_temp(path):
            data = []
            with open_file(path, 'r') as f:
                for key in self.input_key:
                    data.append(self.read_image(f, key)[None])
            data = np.concatenate(data, axis=0)

            tmp_path = os.path.splitext(path)[0] + '_tmp.h5'
            with open_file(tmp_path, 'w') as f:
                f.create_dataset(out_key, data=data, chunks=True)

            return tmp_path

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            tmp_files = list(tqdm(tp.map(_write_temp, input_files), total=len(input_files)))

        return tmp_files

    # note: doesn't work for n5
    def clear_tmp_files(self, tmp_files):
        for path in tmp_files:
            os.remove(path)

    def run(self, input_files, output_files, n_jobs=1,
            n_threads=None, mem_limit=None):

        if n_threads is not None:
            os.environ['LAZYFLOW_THREADS'] = str(n_threads)
        if mem_limit is not None:
            os.environ['LAZYFLOW_TOTAL_RAM_MB'] = str(mem_limit)

        # if we have multiple dataset keys as input, we need to write a temporary stack
        write_stack = isinstance(self.input_key, (list, tuple))
        if write_stack:
            input_key = 'data'
            input_files = self.write_temporary_stack(input_files, input_key, n_jobs)
        else:
            input_key = self.input_key

        _predict = partial(self.predict_images, input_key=input_key)
        _save = partial(self.save_prediction, key=input_key)
        job_files = files_to_jobs(n_jobs, input_files)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_predict, job_files), total=len(job_files)))

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_save, input_files, output_files),
                      total=len(input_files)))

        if write_stack:
            self.clear_tmp_files(input_files)
