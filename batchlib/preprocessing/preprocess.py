import os
from concurrent import futures
from functools import partial

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import barrel_correction, open_file

DEFAULT_CHANNEL_NAMES = ['DAPI', 'WF_GFP', 'TRITC']
DEFAULT_VIEWER_SETTINGS = {'DAPI': {'color': 'Blue'},
                           'WF_GFP': {'color': 'Green'},
                           'TRITC': {'color': 'Red'}}


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
    def validate_barrel_corector_inputs(path, key):
        if path is None:
            return None, None
        if key is None:
            raise ValueError("Expect barrel corrector path AND key")
        if not isinstance(key, (str, list, tuple)):
            raise ValueError("Expect key to be one of str, list or tuple, not %s" % type(key))
        with open_file(path, 'r') as f:
            if isinstance(key, str):
                found = key in f
            else:
                found = all(kk in f for kk in key)
        if not found:
            raise ValueError("Could not find barrel corrector keys")
        return path, key

    def __init__(self, channel_names=DEFAULT_CHANNEL_NAMES,
                 viewer_settings=DEFAULT_VIEWER_SETTINGS, output_ext='.h5',
                 barrel_corrector_path=None, barrel_corrector_key=None,
                 **super_kwargs):
        if len(channel_names) != 3:
            raise ValueError("Expected 3 channels, got %i" % len(channel_names))

        self.channel_names = channel_names
        channel_names_ = ['raw'] + channel_names
        channel_dims = [3, 2, 2, 2]

        (self.barrel_corrector_path,
         self.barrel_corrector_key) = self.validate_barrel_corector_inputs(barrel_corrector_path,
                                                                           barrel_corrector_key)

        super().__init__(input_pattern='*.tiff', output_ext=output_ext,
                         output_key=channel_names_, output_ndim=channel_dims,
                         viewer_settings=viewer_settings, **super_kwargs)

    def _reorder(self, im, reorder):
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

    def preprocess_image(self, in_path, out_path,
                         barrel_corrector, reorder, write_stack):
        im = imageio.volread(in_path)
        im = np.asarray(im)

        n_channels = im.shape[0]
        if reorder is not None:
            assert n_channels == len(reorder), "Expect inputs to have %i channels, got %i" % (len(reorder), n_channels)
            im = self._reorder(im, reorder)

        # apply barrel correction if given
        im_corrected = None
        if barrel_corrector is not None:
            im_corrected = barrel_correction(im, *barrel_corrector)

        with open_file(out_path, 'a') as f:

            # note: we store the uncorrected data in raw for now,
            # because we use this for the instance segmentation pipeline
            if write_stack:
                self.write_result(f, 'raw', im)

            if im_corrected is not None and write_stack:
                self.write_result(f, 'corrected', im_corrected)

            for chan_id, key in enumerate(self.channel_names):

                # save the raw image channel
                chan = im[chan_id]
                self.write_result(f, key, chan)

                # save the corrceted image channel
                if im_corrected is not None:
                    # get the settings for this image channel
                    this_settings = self.viewer_settings[key].copy()
                    this_settings.update({'visible': False})

                    chan = im_corrected[chan_id]
                    self.write_result(f, key + '_corrected', chan, settings=this_settings)

    def load_barrel_corrector(self):
        if self.barrel_corrector_path is None:
            return None

        with open_file(self.barrel_corrector_path, 'r') as f:
            if isinstance(self.barrel_corrector_key, str):
                barrel_corrector = f[self.barrel_corrector_key][:]
            else:
                barrel_corrector = tuple(f[key][:] for key in self.barrel_corrector_key)

        return barrel_corrector

    # TODO we only need write stack if we have ilastik in the pipeline.
    # once we get rid of it, we should set the default to False!
    def run(self, input_files, output_files, reorder=None,
            write_stack=True, n_jobs=1):

        if reorder is not None:
            if not isinstance(reorder, (list, tuple)):
                raise ValueError("Invalid reorder parameter, expect list or tuple, not %s" % type(reorder))

            n_exp = len(self.channel_names)
            n_channels = sum(1 if reord != -1 else 0 for reord in reorder)
            if n_channels != n_exp:
                raise ValueError("Invalid number of channels after reordering, expect %i, got %i" % (n_exp,
                                                                                                     n_channels))
        barrel_corrector = self.load_barrel_corrector()
        _preprocess = partial(self.preprocess_image,
                              barrel_corrector=barrel_corrector,
                              reorder=reorder,
                              write_stack=write_stack)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_preprocess, input_files, output_files), total=len(input_files)))
