import json
import os

from .preprocess import Preprocess


class PreprocessTelesto(Preprocess):
    reader_function = 'imread'
    channel_first = False
    semantic_viewer_settings = {'nuclei': {'color': 'Blue', 'visible': True},
                                'serum_IgA': {'color': 'Red', 'visible': True},
                                'serum_IgG': {'color': 'Green', 'visible': True}}

    @classmethod
    def from_folder(cls, input_folder, output_ext=None,
                    barrel_corrector_path=None, **super_kwargs):

        # load channel name -> semantic name mapping
        mapping_file = os.path.join(input_folder, 'channel_mapping.json')
        if not os.path.exists(mapping_file):
            raise ValueError(f"The mapping file {mapping_file} does not exist.")
        with open(mapping_file, 'r') as f:
            channel_mapping = json.load(f)

        viewer_settings = {}
        for semantic_name in channel_mapping:
            if semantic_name is None:
                continue

            this_settings = None
            for name, settings in cls.semantic_viewer_settings.items():
                if semantic_name.startswith(name):
                    this_settings = settings
                    break

            if this_settings is None:
                raise ValueError(f"Did not find a matching semantic channel for {semantic_name}")
            viewer_settings[semantic_name] = this_settings

        channel_names = channel_mapping
        channel_mapping = {name: name for name in channel_names}
        return cls(channel_names, channel_mapping, viewer_settings,
                   output_ext=output_ext,
                   barrel_corrector_path=barrel_corrector_path,
                   input_pattern='*.png',
                   **super_kwargs)
