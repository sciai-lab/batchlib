import json
import os
from concurrent import futures
from functools import partial
from glob import glob

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..config import get_default_extension
from ..util import barrel_correction, open_file


def parse_channel_names(input_name):
    """ Parse channel names from input name
    """

    # make sure this is just the filename
    file_name = os.path.split(input_name)[1]

    # parse the channel names from the file name
    channel_str = '_'.join(file_name.split('_')[3:-1])
    assert channel_str.startswith('Channel')
    channel_str = channel_str.lstrip('Channel')
    channel_names = channel_str.split(',')
    return tuple(channel_names)


# TODO for this to work, we need to move the barrel correctors in their
# own sub-directory and add a global attribute 'image_shape' to them
def find_barrel_corrector(barrel_corrector_root, image_shape):
    barrel_correctors = glob(os.path.join(barrel_corrector_root, '*.h5'))
    for corrector_path in barrel_correctors:
        with open_file(corrector_path, 'r') as f:
            corrector_shape = f.attrs['image_shape']
        if corrector_shape == image_shape:
            return corrector_path
    raise RuntimeError(f"Could not find barrel corrector for image shape {image_shape}")


class Preprocess(BatchJobOnContainer):
    """ Preprocess folder with tifs from high-throughput experiment
    """
    semantic_viewer_settings = {'nuclei': {'color': 'Blue', 'visible': True},
                                'serum': {'color': 'Green', 'visible': True},
                                'marker': {'color': 'Red', 'visible': True},
                                'cells': {'color': 'Gray', 'visible': False}}

    @staticmethod
    def validate_barrel_corrector(path, channel_names, channel_mapping):
        if path is None:
            return
        with open_file(path, 'r') as f:
            channels_found = [name in f for name in channel_names if channel_mapping[name] is not None]
        if not all(channels_found):
            missing_channels = ' '.join(chan for chan, found in zip(channel_names, channels_found) if not found)
            raise ValueError("Could not find all channel names in the barrel corrector file: %s" % missing_channels)

    @classmethod
    def from_folder(cls, input_folder, output_ext=None,
                    barrel_corrector_path=None, **super_kwargs):

        # load channel name -> semantic name mapping
        mapping_file = os.path.join(input_folder, 'channel_mapping.json')
        if not os.path.exists(mapping_file):
            raise ValueError("The input folder %s does not contain channel_mapping.json" % mapping_file)
        with open(mapping_file, 'r') as f:
            channel_mapping = json.load(f)

        viewer_settings = {}
        for chan_name, semantic_name in channel_mapping.items():
            if semantic_name is None:
                continue

            this_settings = None
            for name, settings in cls.semantic_viewer_settings.items():
                if semantic_name.startswith(name):
                    this_settings = settings
                    break

            if this_settings is None:
                raise ValueError("Did not find a matching semantic channel for %s" % semantic_name)
            viewer_settings[semantic_name] = this_settings

        # parse the channel names and make sure they match with the mapping
        pattern = os.path.join(input_folder, '*.tiff')
        input_files = glob(pattern)
        channel_names = parse_channel_names(input_files[0])
        if len(set(channel_names) - set(channel_mapping.keys())) != 0:
            raise ValueError("Channel mapping and channel names does not match")

        # check that all files have the same channels
        if any(parse_channel_names(path) != channel_names for path in input_files[1:]):
            raise ValueError("Channel names are not consistent for all files")

        return cls(channel_names, channel_mapping, viewer_settings,
                   output_ext=output_ext,
                   barrel_corrector_path=barrel_corrector_path, **super_kwargs)

    def __init__(self, channel_names, channel_mapping, viewer_settings,
                 output_ext=None, barrel_corrector_path=None,
                 **super_kwargs):

        output_ext = get_default_extension() if output_ext is None else output_ext

        self.validate_barrel_corrector(barrel_corrector_path, channel_names, channel_mapping)
        self.barrel_corrector_path = barrel_corrector_path

        self.channel_mapping = channel_mapping
        self.channel_names = channel_names

        channel_out_names = list(viewer_settings.keys())
        if self.barrel_corrector_path is not None:
            channel_out_names += [chan_name + '_corrected' for chan_name in channel_out_names]
        channel_out_dims = [2] * len(channel_out_names)

        # add the channel information to the viewer settings, so
        # we can keep track of the original channel names later
        for name, semantic_name in self.channel_mapping.items():
            if semantic_name is None:
                continue
            viewer_settings[semantic_name].update({'channel_information': name})

        super().__init__(input_pattern='*.tiff', output_ext=output_ext,
                         output_key=channel_out_names, output_ndim=channel_out_dims,
                         viewer_settings=viewer_settings, **super_kwargs)

    def preprocess_image(self, in_path, out_path, barrel_corrector):
        im = imageio.volread(in_path)
        im = np.asarray(im)

        # note: we don't delay the keyboard interruption here, because this code
        # is usually not executed in the main thread
        with open_file(out_path, 'a') as f:

            for chan_id, name in enumerate(self.channel_names):

                semantic_name = self.channel_mapping[name]
                if semantic_name is None:
                    continue

                this_settings = self.viewer_settings[semantic_name].copy()
                if barrel_corrector is not None:
                    this_settings.update({'visible': False})

                # save the raw image channel
                chan = im[chan_id]
                self.write_image(f, semantic_name, chan, settings=this_settings)

                # apply and save the barrel corrected channel,
                # if we have a barrel corrector
                if barrel_corrector is not None:
                    this_corrector = barrel_corrector[name]
                    this_settings.update({'visible': True})
                    chan = barrel_correction(chan, *this_corrector)

                    # get the settings for this image channel
                    self.write_image(f, semantic_name + '_corrected', chan,
                                     settings=this_settings)

    def load_barrel_corrector(self):
        if self.barrel_corrector_path is None:
            return None

        barrel_corrector = {}
        with open_file(self.barrel_corrector_path, 'r') as f:
            for name in self.channel_names:

                if self.channel_mapping[name] is None:
                    continue

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
