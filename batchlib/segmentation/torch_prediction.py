import os
from math import ceil

import numpy as np
import torch
from tqdm import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.segmentation.unet import UNet2D
from batchlib.util import open_file, files_to_jobs, write_viewer_attributes, standardize


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

    def predict_images(self, in_batch, out_batch, model,
                       default_normalize, device, threshold_channels):
        inputs = []
        for in_path in in_batch:
            with open_file(in_path, 'r') as f:
                if self.input_channel is None:
                    im = f[self.input_key][:]
                else:
                    im = f[self.input_key][self.input_channel]

                assert im.ndim == 2
                # TODO this should not be hard-coded to the model class, but passed as an extra parameter.
                # for now, we only run UNet2D anyways, so it doesn't matter
                squeeze_z = False
                if isinstance(model, UNet2D):
                    # add batch, channel and Z axes required fo UNet2d
                    im = im[None, None, None]
                    squeeze_z = True
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
        if squeeze_z:
            prediction = prediction.squeeze(2)

        for out_path, pred in zip(out_batch, prediction):
            assert pred.shape[0] == len(self._output_exp_key)
            with open_file(out_path, 'a') as f:
                for channel_id, (key, channel) in enumerate(zip(self._output_exp_key, pred)):
                    threshold = threshold_channels.get(channel_id, None)
                    if threshold is not None:
                        channel = (channel > threshold).astype('uint8')
                    ds = f.require_dataset(key, shape=channel.shape, compression='gzip',
                                           dtype=channel.dtype)
                    ds[:] = channel
                    write_viewer_attributes(ds, channel, 'Gray')

    # load from pickled model or from state dict
    def load_model(self, device):
        if self.model_class is None:
            model = torch.load(self.model_path, map_location=device)
        else:
            model = self.model_class(**self.model_kwargs)
            state = torch.load(self.model_path, map_location=device)
            model.load_state_dict(state)
        model.eval()
        return model

    # threshold channels is a dict mapping channel ids that should be thresholded to the threshold value
    def run(self, input_files, output_files, gpu_id=None, batch_size=1,
            normalize=standardize, threshold_channels={}):

        with torch.no_grad():
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                device = torch.device(0)
            else:
                device = torch.device('cpu')
            model = self.load_model(device)
            model.to(device)

            n_batches = int(ceil(len(input_files) / float(batch_size)))
            input_batches = files_to_jobs(n_batches, input_files)
            output_batches = files_to_jobs(n_batches, output_files)

            for in_batch, out_batch in tqdm(zip(input_batches, output_batches), total=len(input_batches)):
                self.predict_images(in_batch, out_batch, model, normalize, device, threshold_channels)