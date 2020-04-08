import os
from concurrent import futures

from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file


# TODO (or rather to premature optimize, this is fast enough on single gpu for now!)
# - implement multi gpu support, would probably need to do IPC through files or
#    call this script with inputs / outputs in subprocess, it Deadlocked when running
#    from one process pool
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

    def run(self, input_files, output_files, gpu_id):

        if gpu_id is None:
            # need to do this for the conda tensorflow gpu verion
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        model = StarDist2D(None, name=self.model_name, basedir=self.model_root)
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            self.segment_image(in_path, out_path, model)
