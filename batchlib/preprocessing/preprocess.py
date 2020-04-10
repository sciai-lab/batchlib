from concurrent import futures
from functools import partial

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJob
from ..util import barrel_correction, open_file, write_viewer_attributes, set_skip

DEFAULT_CHANNEL_NAMES = ['DAPI', 'WF_GFP', 'TRITC']


# TODO
# - something still locks gil, probably h5py. should try processs pool
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

    def _reorder(self, im):
        im_new = np.zeros_like(im)
        im_new[0] = im[0]
        im_new[1] = im[2]
        im_new[2] = im[1]
        return im_new

    def preprocess_image(self, in_path, out_path, barrel_corrector, reorder, keep_raw):
        im = imageio.volread(in_path)
        im = np.asarray(im)

        n_channels = im.shape[0]
        # if reorder is None, try to get it from the data
        if reorder is None:
            if n_channels == 3:
                reorder = True
            elif n_channels == 4:
                reorder = False
            else:
                raise RuntimeError("Expect inputs to have %i chanels" % n_channels)

        # for new iteration of the data, we need to reorder the channels
        # and we don't have the garbage channel
        if reorder:
            assert n_channels == 3, "Expect inputs to have 3 channels, got %i" % n_channels
            im = self._reorder(im)
        else:
            # get rid of garbage channels
            assert n_channels == 4, "Expect inputs to have 4 channels, got %i" % n_channels
            im = im[:3]

        # TODO use the proper flat-field-correction (we will need add. offset parameter for that)
        # apply barrel correction
        im_raw = None
        if barrel_corrector is not None:
            if keep_raw:
                im_raw = im.copy()
            im = barrel_correction(im, barrel_corrector)

        colors = ['Red', 'Green', 'Blue']
        with open_file(out_path, 'a') as f:
            # TODO try to have the raw data as region references to the individual channels
            ds = f.require_dataset('raw', shape=im.shape, dtype=im.dtype,
                                   compression='gzip')
            ds[:] = im
            set_skip(ds)

            for chan_id, (color, key) in enumerate(zip(colors, self.channel_names)):
                chan = im[chan_id]
                ds = f.require_dataset(key, shape=chan.shape, dtype=chan.dtype,
                                       compression='gzip')
                ds[:] = im[chan_id]
                write_viewer_attributes(ds, chan, color)

                if im_raw is not None:
                    chan = im_raw[chan_id]
                    ds = f.require_dataset(key + '_raw', shape=chan.shape, dtype=chan.dtype,
                                           compression='gzip')
                    ds[:] = chan
                    write_viewer_attributes(ds, chan, color, visible=False)

    def run(self, input_files, output_files, reorder=None, barrel_corrector=None,
            keep_raw=True, n_jobs=1):

        _preprocess = partial(self.preprocess_image,
                              barrel_corrector=barrel_corrector,
                              reorder=reorder,
                              keep_raw=keep_raw)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_preprocess, input_files, output_files), total=len(input_files)))
