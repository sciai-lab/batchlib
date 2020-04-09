import os
import json
import subprocess
import sys

from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, write_viewer_attributes


# TODO (or rather to premature optimize, this is fast enough on single gpu for now!)
# - implement multi gpu support, would probably need to do IPC through files or
#    call this script with inputs / outputs in subprocess, it Deadlocked when running
#    from one process pool
# - or is there something like torch.data_parallel one could use?
class StardistPrediction(BatchJobOnContainer):
    """
    """
    def __init__(self, model_root, model_name,
                 input_key, output_key,
                 input_channel=None, input_pattern='*.h5'):
        self.input_channel = input_channel
        input_ndim = 2 if self.input_channel is None else 3
        super().__init__(input_pattern,
                         input_key=input_key, output_key=output_key,
                         input_ndim=input_ndim, output_ndim=2)

        self.model_root = model_root
        self.model_name = model_name
        self.runners = {'default': self.run}

    def segment_image(self, in_path, out_path, model):
        with open_file(in_path, 'r') as f:
            if self.input_channel is None:
                im = f[self.input_key][:]
            else:
                im = f[self.input_key][self.input_channel]

        im = normalize(im, 1, 99.8)
        labels, _ = model.predict_instances(im)
        labels = labels.astype('uint32')
        with open_file(out_path, 'a') as f:
            ds = f.require_dataset(self.output_key, shape=labels.shape, compression='gzip',
                                   dtype=labels.dtype)
            ds[:] = labels
            write_viewer_attributes(ds, labels, 'Glasbey')

    def run(self, input_files, output_files, gpu_id=None, pybin=None):

        if pybin is not None:
            print("Run stardist prediction in subprocess with bin", pybin)
            build_kwargs = {'model_root': self.model_root, 'model_name': self.model_name,
                            'input_key': self.input_key, 'output_key': self.output_key,
                            'input_channel': self.input_channel, 'input_pattern': self.input_pattern}
            build_kwargs = json.dumps(build_kwargs)

            run_kwargs = {'input_files': input_files, 'output_files': output_files, 'gpu_id': gpu_id}
            run_kwargs = json.dumps(run_kwargs)

            cmd = [pybin, __file__, build_kwargs, run_kwargs]
            subprocess.run(cmd)

        if gpu_id is None:
            # need to do this for the conda tensorflow cpu version
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        model = StarDist2D(None, name=self.model_name, basedir=self.model_root)
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            self.segment_image(in_path, out_path, model)


if __name__ == '__main__':
    build_kwargs, run_kwargs = sys.argv[1], sys.argv[2]

    build_kwargs = json.loads(build_kwargs)
    run_kwargs = json.loads(run_kwargs)

    job = StardistPrediction(**build_kwargs)
    job.run(**run_kwargs)
