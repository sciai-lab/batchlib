import os
import subprocess
from concurrent import futures
from tqdm import tqdm

from ..base import BatchJob
from ..util import open_file, files_to_jobs


# TODO
# - ilastik saves files with 'w'. this should be changed to 'a', and then we can write directly
# - subprocess doesn't seem to lift gil fully, could use ProcessPool, but this doesn't work for some reason
class IlastikPrediction(BatchJob):
    """
    """

    def __init__(self, ilastik_bin, ilastik_project,
                 input_key, output_key,
                 input_ndim=None, output_ndim=None):
        super().__init__(input_key, output_key, input_ndim, output_ndim)

        self.bin = ilastik_bin
        self.project = ilastik_project

        self.runners = {'local': self.run,
                        'slurm': self.run_slurm}

    def predict_images(self, input_files):
        inputs = ['%s/%s' % (inp, self.input_key) for inp in input_files]
        cmd = [self.bin, '--headless', '--readonly', '--project=%s' % self.project]
        cmd.extend(inputs)
        subprocess.run(cmd, check=True)

    def save_prediction(self, in_path, out_path):
        tmp_path = in_path[:-3] + '-raw_Probabilities.h5'
        tmp_key = 'exported_data'

        # load the back-ground and boundary channels
        with open_file(tmp_path, 'r') as f:
            ds = f[tmp_key]
            data = ds[:]

        with open_file(out_path, 'a') as f:
            ds = f.require_dataset(self.output_key, shape=data.shape,
                                   dtype='float32', compression='gzip')
            ds[:] = data

        # clean up
        os.remove(tmp_path)

    def run(self, input_files, output_files, n_jobs=1,
            n_threads=None, mem_limit=None):

        if n_threads is not None:
            os.environ['LAZYFLOW_THREADS'] = str(n_threads)
        if mem_limit is not None:
            os.environ['LAZYFLOW_TOTAL_RAM_MB'] = str(mem_limit)

        # run with multiple jobs
        if n_jobs > 1:

            job_files = files_to_jobs(n_jobs, input_files)
            # TODO would be better to run with a process pool
            # but somehow this gets stuck
            # with futures.ProcessPoolExecutor(n_jobs) as pp:
            with futures.ThreadPoolExecutor(n_jobs) as tp:
                list(tqdm(tp.map(self.predict_images, job_files), total=len(job_files)))

            with futures.ThreadPoolExecutor(n_jobs) as tp:
                list(tqdm(tp.map(self.save_prediction, input_files, output_files),
                          total=len(input_files)))

        # run with single job
        else:
            # predict the images
            self.predict_images(input_files)

            # load predictions and save to the output
            for in_path, out_path in tqdm(zip(input_files, output_files)):
                self.save_prediction(in_path, out_path)

    # this would implement prediction on slurm cluster
    def run_slurm(self, input_files, output_files, n_jobs=1,
                  n_threads=None, mem_limit=None):
        raise NotImplementedError
