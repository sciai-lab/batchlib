import os

from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file


# TODO
# - implement on gpu
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
        with open_file(out_path, 'a') as f:
            ds = f.require_dataset(self.output_key, shape=labels.shape, compression='gzip',
                                   dtype=labels.dtype)
            ds[:] = labels

    def _run_gpu(self, input_files, output_files, gpu_ids):
        raise NotImplementedError

    def _run_cpu(self, input_files, output_files):
        # need to do this for the conda tensorflow version
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        model = StarDist2D(None, name=self.model_name, basedir=self.model_root)
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            self.segment_image(in_path, out_path, model)

    def run(self, input_files, output_files, gpu_ids=None):
        # gpu-ids is None -> run on the cpu
        if gpu_ids is None:
            self._run_cpu(input_files, output_files)
        # otherwise, run on gpu
        else:
            self._run_gpu(input_files, output_files, gpu_ids)
