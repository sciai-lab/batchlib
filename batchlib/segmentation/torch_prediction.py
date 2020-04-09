import os
import numpy as np

import torch
from tqdm import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.segmentation.unet import UNet2D
from batchlib.util import open_file, files_to_jobs
from batchlib.util.image import normalize as default_normalize


# TODO
# - to optimize gpu throughput further could use torch.parallel / torch.data_parallel
#   or dask.delayed to parallelize the input loading and output writing
class TorchPrediction(BatchJobOnContainer):
    """
    """

    def __init__(self, input_key, output_key, model_path,
                 model_class=None, model_kwargs={},
                 input_channel=None, input_pattern='*.h5'):
        self.input_channel = input_channel
        input_ndim = 2 if self.input_channel is None else 3
        super().__init__(input_pattern,
                         input_key=input_key, output_key=output_key,
                         input_ndim=input_ndim, output_ndim=2)

        self.model_path = model_path
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.runners = {'default': self.run}

    def predict_images(self, in_batch, out_batch, model, default_normalize, device):
        inputs = []
        for in_path in in_batch:
            with open_file(in_path, 'r') as f:
                if self.input_channel is None:
                    im = f[self.input_key][:]
                else:
                    im = f[self.input_key][self.input_channel]

                assert im.ndim == 2
                if isinstance(model, UNet2D):
                    # add batch, channel and Z axes required fo UNet2d
                    im = im[None, None, None]
                else:
                    # add batch and channel axis
                    im = im[None, None]

                # normalize the image
                im = default_normalize(im)
                inputs.append(im)

        inputs = np.concatenate(inputs, axis=0)
        inputs = torch.from_numpy(inputs).float().to(device)
        prediction = model(inputs)
        prediction = prediction.cpu().numpy()

        for out_path, pred in zip(out_batch, prediction):
            assert pred.shape[0] == len(self._output_exp_key)
            with open_file(out_path, 'a') as f:
                for key, channel in zip(self._output_exp_key, pred):
                    ds = f.require_dataset(key, shape=channel.shape, compression='gzip',
                                           dtype=channel.dtype)
                    ds[:] = channel

    # load from pickled model or from state dict
    def load_model(self, device):
        if self.model_class is None:
            model = torch.load(self.model_path, map_location=device)
        else:
            model = self.model_class(**self.model_kwargs)
            state = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(state)
        model.eval()
        return model

    def run(self, input_files, output_files, gpu_id=None, batch_size=1, normalize=default_normalize):

        with torch.no_grad():
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                device = torch.device(0)
            else:
                device = torch.device('cpu')
            model = self.load_model(device)
            model.to(device)

            #input_batches, output_batches = files_to_jobs()
            # TODO: just for testing, use files_to_jobs() semantics when it's ready
            input_batches, output_batches = input_files, output_files

            for in_batch, out_batch in tqdm(zip(input_batches, output_batches), total=len(input_files)):
                self.predict_images(in_batch, out_batch, model, normalize, device)
