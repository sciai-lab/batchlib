import os
from tqdm import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, normalize_percentile


class StardistPrediction(BatchJobOnContainer):
    """
    """
    script = __file__

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

    def segment_image(self, in_path, out_path, model):
        with open_file(in_path, 'r') as f:
            if self.input_channel is None:
                im = f[self.input_key][:]
            else:
                im = f[self.input_key][self.input_channel]

        im = normalize_percentile(im, 1, 99.8)
        labels, _ = model.predict_instances(im)
        labels = labels.astype('uint32')
        with open_file(out_path, 'a') as f:
            self.write_result(f, self.output_key, labels)

    def run(self, input_files, output_files, gpu_id=None):

        if gpu_id is None:
            # need to do this for the conda tensorflow cpu version
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # we only import this when necessary, it pulls in tf with all it's ugliness ...
        from stardist.models import StarDist2D
        model = StarDist2D(None, name=self.model_name, basedir=self.model_root)
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            self.segment_image(in_path, out_path, model)
