from concurrent import futures
from functools import partial
from tqdm import tqdm

from ..analysis.cell_level_analysis import _get_bg_correction_dict
from ..base import BatchJobOnContainer
from ..util import open_file, in_file_to_image_name


class WriteBackgroundSubtractedImages(BatchJobOnContainer):
    def __init__(self, background_dict, table_path):
        self.background_dict = background_dict
        self.table_path = table_path

        self.channel_names = list(background_dict.keys())

        self.output_keys = [key + '_background_subtracted' for key in background_dict.keys()]
        assert len(self.channel_names) == len(self.output_keys)

        super().__init__(input_key=self.channel_names,
                         input_format=['image'] * len(self.channel_names),
                         output_key=self.output_keys,
                         output_format=['image'] * len(self.output_keys))

    def write_bg_image(self, in_file, out_file, background_dict):
        images = []

        with open_file(in_file, 'r') as f:
            for channel_name in self.channel_names:
                im = self.read_image(f, channel_name)
                image_name = in_file_to_image_name(in_file)
                bg_val = background_dict[channel_name][image_name]
                images.append(im - bg_val)

        with open_file(out_file, 'a') as f:
            for out_key, im in zip(self.output_keys, images):
                self.write_image(f, out_key, im)

    def get_actual_bacground_dict(self, input_files):
        actual_bg_dict = {}
        for channel_name, bg_val in self.background_dict.items():
            bg_col_name = f'{channel_name}_median'
            actual_bg_dict[channel_name] = _get_bg_correction_dict(self.table_path, bg_val,
                                                                   bg_col_name, input_files)
        return actual_bg_dict

    def run(self, input_files, output_files, n_jobs=1):

        background_dict = self.get_actual_bacground_dict(input_files)
        _write_im = partial(self.write_bg_image, background_dict=background_dict)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_write_im, input_files, output_files),
                      desc='write background images',
                      total=len(input_files)))
