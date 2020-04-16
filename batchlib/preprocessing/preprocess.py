import json
import os
from concurrent import futures
from functools import partial
from glob import glob

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
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


class Preprocess(BatchJobOnContainer):
    """ Preprocess folder with tifs from high-throughput experiment
    """
    semantic_viewer_settings = {'nuclei': {'color': 'Blue'},
                                'serum': {'color': 'Green'},
                                'marker': {'color': 'Red'}}

    @staticmethod
    def validate_barrel_corector(path, channel_names):
        if path is None:
            return
        with open_file(path, 'r') as f:
            channels_found = [name in f for name in channel_names]
        if not all(channels_found):
            raise ValueError("Could not find all channel names in the barrel corrector file")

    @classmethod
    def from_folder(cls, input_folder, output_ext='.h5',
                    barrel_corrector_path=None, **super_kwargs):

        # load channel name -> semantic name mapping
        mapping_file = os.path.join(input_folder, 'channel_mapping.json')
        if not os.path.exists(mapping_file):
            raise ValueError("The input folder %s does not contain channel_mapping.json" % mapping_file)
        with open(mapping_file, 'r') as f:
            channel_mapping = json.load(f)

        # validate the channel mapping:
        mapping_values = set(val for val in channel_mapping.values() if val is not None)
        # make sure that we have exaclty three valid channel names
        if len(mapping_values) != 3:
            raise ValueError("Invalid channel mapping")
        # make sure that we map to the correct names
        if len(set(cls.semantic_viewer_settings.keys()) - mapping_values) != 0:
            raise ValueError("Invalid channel mapping")

        # parse the channel names and make sure they match with the mapping
        pattern = os.path.join(input_folder, '*.tiff')
        input_files = glob(pattern)
        channel_names = parse_channel_names(input_files[0])
        if len(set(channel_names) - set(channel_mapping.keys())) != 0:
            raise ValueError("Channel mapping and channel names does not match")

        # check that all files have the same channels
        if any(parse_channel_names(path) != channel_names for path in input_files[1:]):
            raise ValueError("Channel names are not consistent for all files")

        return cls(channel_names, channel_mapping,
                   output_ext=output_ext,
                   barrel_corrector_path=barrel_corrector_path, **super_kwargs)

    def __init__(self, channel_names, channel_mapping,
                 output_ext='.h5', barrel_corrector_path=None,
                 **super_kwargs):

        self.validate_barrel_corector(barrel_corrector_path, channel_names)
        self.barrel_corrector_path = barrel_corrector_path

        self.channel_mapping = channel_mapping
        self.channel_names = channel_names

        channel_out_names = list(self.semantic_viewer_settings.keys())
        if self.barrel_corrector_path is not None:
            channel_out_names += [chan_name + '_corrected' for chan_name in channel_out_names]
        channel_out_dims = [2] * len(channel_out_names)

        # add the channel information to the viewer settings, so
        # we can keep track of the original channel names later
        viewer_settings = self.semantic_viewer_settings
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

        with open_file(out_path, 'a') as f:

            for chan_id, name in enumerate(self.channel_names):

                semantic_name = self.channel_mapping[name]
                if semantic_name is None:
                    continue

                # save the raw image channel
                chan = im[chan_id]
                self.write_result(f, semantic_name, chan)

                # apply and save the barrel corrected channel,
                # if we have a barrel corrector
                if barrel_corrector is not None:
                    this_corrector = barrel_corrector[name]
                    chan = barrel_correction(chan, *this_corrector)

                    # get the settings for this image channel
                    this_settings = self.viewer_settings[semantic_name].copy()
                    this_settings.update({'visible': False})
                    self.write_result(f, semantic_name + '_corrected', chan, settings=this_settings)

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
