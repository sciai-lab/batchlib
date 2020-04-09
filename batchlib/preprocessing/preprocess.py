from concurrent import futures
from functools import partial

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJob
from ..util import barrel_correction, open_file, write_viewer_attributes

DEFAULT_CHANNEL_NAMES = ['DAPI', 'WF_GFP', 'TRITC']


class Preprocess(BatchJob):

    def __init__(self, channel_names=DEFAULT_CHANNEL_NAMES, output_ext='.h5'):
        if len(channel_names) != 3:
            raise ValueError("Expected 3 channels, got %i" % len(channel_names))
        self.channel_names = channel_names
        channel_names_ = ['raw'] + channel_names
        channel_dims = [3, 2, 2, 2]
        super().__init__(input_pattern='*.tiff', output_ext=output_ext,
                         output_key=channel_names_, output_ndim=channel_dims)
        self.runners = {'default': self.run}

    def _reorder(im):
        im_new = np.zeros_like(im)
        im_new[0] = im[0]
        im_new[1] = im[2]
        im_new[2] = im[1]
        return im_new

    def preprocess_image(self, in_path, out_path, barrel_corrector, reorder):
        im = imageio.volread(in_path)
        im = np.asarray(im)

        # for new iteration of the data, we need to reorder the channels
        # and we don't have the garbage channel
        if reorder:
            assert im.shape[0] == 3, "Expect inputs to have 3 channels, got %i" % im.shape[0]
            im = self._reorder(im)
        else:
            # get rid of garbage channels
            assert im.shape[0] == 4, "Expect inputs to have 4 channels, got %i" % im.shape[0]
            im = im[:3]

        # TODO save the corrected and the not-corrected version
        # TODO use the proper flat-field-correction
        # apply barrel correction
        if barrel_corrector is not None:
            im = barrel_correction(im, barrel_corrector)

        with open_file(out_path, 'a') as f:
            # TODO try to have the raw data as region references to the individual channels
            ds = f.require_dataset('raw', shape=im.shape, dtype=im.dtype,
                                   compression='gzip')
            ds[:] = im
            for key, chan in zip(self.channel_names, im):
                ds = f.require_dataset(key, shape=chan.shape, dtype=chan.dtype,
                                       compression='gzip')
                ds[:] = chan
                write_viewer_attributes(ds, chan, 'raw')

    def run(self, input_files, output_files,
            reorder=True, barrel_corrector_path=None, barrel_corrector_key='data',
            n_jobs=1):

        if barrel_corrector_path is None:
            barrel_corrector = None
        else:
            with open_file(barrel_corrector_path, 'r') as f:
                barrel_corrector = f[barrel_corrector_key][:]

        _preprocess = partial(self.preprocess_image, barrel_corrector=barrel_corrector, reorder=reorder)
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_preprocess, input_files, output_files), total=len(input_files)))
