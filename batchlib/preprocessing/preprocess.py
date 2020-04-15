import os
from concurrent import futures
from functools import partial

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import barrel_correction, open_file


def get_channel_settings(input_name):
    """ Match the channel names to expected channel order and
    viewer settings from file name.

    Note: if multiple channel names match, will match to the first.
    """

    # make sure this is just the filename
    file_name = os.path.split(input_name)[1]

    # parse the channel names from the file name
    channel_str = '_'.join(file_name.split('_')[3:-1])
    assert channel_str.startswith('Channel')
    channel_str = channel_str.lstrip('Channel')
    channel_names_raw = channel_str.split(',')

    reorder = []
    viewer_settings = {}

    have_dapi, have_wf, have_tritc = False, False, False

    # match the channel names to our defaults
    for name in channel_names_raw:

        if name.startswith('DAPI'):
            if have_dapi:
                reorder.append(-1)
                continue
            else:
                reorder.append(0)
                have_dapi = True
                color = 'Blue'

        if name.startswith('WF'):
            if have_wf:
                reorder.append(-1)
                continue
            else:
                reorder.append(1)
                have_wf = True
                color = 'Green'

        if name.startswith('TRITC'):
            if have_tritc:
                reorder.append(-1)
                continue
            else:
                reorder.append(2)
                have_tritc = True
                color = 'Red'

        viewer_settings[name] = {'color': color}

    channel_names = [None] * 3
    for chan_name, reorder_id in zip(channel_names_raw, reorder):
        if reorder_id == -1:
            continue
        channel_names[reorder_id] = chan_name

    assert len(channel_names) == 3
    assert not any(chan_name is None for chan_name in channel_names)
    return channel_names, viewer_settings, reorder


class Preprocess(BatchJobOnContainer):

    @staticmethod
    def validate_barrel_corector(path, channel_names):
        if path is None:
            return
        with open_file(path, 'r') as f:
            channels_found = [name in f for name in channel_names]
        if not all(channels_found):
            raise ValueError("Could not find all channel names in the barrel corrector file")

    @staticmethod
    def validate_reorder(reorder, channel_names):
        if reorder is None:
            return

        if not isinstance(reorder, (list, tuple)):
            raise ValueError("Invalid reorder parameter, expect list or tuple, not %s" % type(reorder))

        n_exp = len(channel_names)
        n_channels = sum(1 if reord != -1 else 0 for reord in reorder)
        if n_channels != n_exp:
            raise ValueError("Invalid number of channels after reordering, expect %i, got %i" % (n_exp,
                                                                                                 n_channels))

    def __init__(self, channel_names, viewer_settings, reorder=None,
                 output_ext='.h5', barrel_corrector_path=None,
                 **super_kwargs):
        if len(channel_names) != 3:
            raise ValueError("Expected 3 channels, got %i" % len(channel_names))

        self.validate_barrel_corector(barrel_corrector_path, channel_names)
        self.barrel_corrector_path = barrel_corrector_path

        self.validate_reorder(reorder, channel_names)
        self.reorder = reorder

        self.channel_names = channel_names

        # we expect duplicated channels when running with barrel correction
        if self.barrel_corrector_path is None:
            expected_channel_names = self.channel_names
        else:
            expected_channel_names = self.channel_names + [chan_name + '_corrected'
                                                           for chan_name in self.channel_names]
        expected_channel_dims = len(expected_channel_names) * [2]

        super().__init__(input_pattern='*.tiff', output_ext=output_ext,
                         output_key=self.channel_names, output_ndim=expected_channel_dims,
                         viewer_settings=viewer_settings, **super_kwargs)

    @staticmethod
    def _reorder(im, reorder):
        im_shape = im.shape[1:]
        n_new_channels = sum(1 if reord != -1 else 0 for reord in reorder)
        new_shape = (n_new_channels,) + im_shape

        im_new = np.zeros(new_shape, dtype=im.dtype)

        chan_id = 0
        for reorder_id in reorder:
            if reorder_id == -1:
                continue
            im_new[chan_id] = im[reorder_id]
            chan_id += 1

        return im_new

    def preprocess_image(self, in_path, out_path, barrel_corrector):
        im = imageio.volread(in_path)
        im = np.asarray(im)

        n_channels = im.shape[0]
        if self.reorder is not None:
            assert n_channels == len(self.reorder), "Expect inputs to have %i channels, got %i" % (len(self.reorder),
                                                                                                   n_channels)
            im = self._reorder(im, self.reorder)

        with open_file(out_path, 'a') as f:

            for chan_id, name in enumerate(self.channel_names):

                # save the raw image channel
                chan = im[chan_id]
                self.write_result(f, name, chan)

                # apply and save the barrel corrected channel,
                # if we have a barrel corrector
                if barrel_corrector is not None:
                    this_corrector = barrel_corrector[name]
                    im_corrected = barrel_correction(im, *this_corrector)

                    # get the settings for this image channel
                    this_settings = self.viewer_settings[name].copy()
                    this_settings.update({'visible': False})

                    chan = im_corrected[chan_id]
                    self.write_result(f, name + '_corrected', chan, settings=this_settings)

    def load_barrel_corrector(self):
        if self.barrel_corrector_path is None:
            return None

        barrel_corrector = {}
        with open_file(self.barrel_corrector_path, 'r') as f:
            for name in self.channel_names:
                ds = f[name]
                corrector = ds[:]
                offset = ds.attrs['offset']
                barrel_corrector[name] = (corrector, offset)

        return barrel_corrector

    def run(self, input_files, output_files, n_jobs=1):

        barrel_corrector = self.load_barrel_corrector()
        _preprocess = partial(self.preprocess_image, barrel_corrector=barrel_corrector)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_preprocess, input_files, output_files), total=len(input_files)))
