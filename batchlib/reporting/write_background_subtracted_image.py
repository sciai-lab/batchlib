from concurrent import futures
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file


class WriteBackgroundSubtractedImages(BatchJobOnContainer):
    def __init__(self, background_dict):
        self.background_dict = background_dict

        self.input_keys = list(background_dict.keys())
        table_keys = [val for val in background_dict.values() if isinstance(val, str)]
        input_keys = self.input_keys + table_keys

        self.output_keys = [key + '_background_subtracted' for key in background_dict.keys()]
        assert len(self.input_keys) == len(self.output_keys)

        super().__init__(input_key=input_keys,
                         input_format=['table'] * len(input_keys),
                         output_key=self.output_keys,
                         output_format=['image'] * len(self.output_keys))

    def write_bg_image(self, in_file, out_file):
        images = []

        with open_file(in_file, 'r') as f:
            for in_key in self.input_keys:
                im = self.read_image(f, in_key)
                bg_val = self.background_dict[in_key]
                if isinstance(bg_val, str):
                    # TODO use Roman's convenience function for this
                    pass
                im -= bg_val
                images.append(im)

        with open_file(out_file, 'a') as f:
            for out_key, im in zip(self.output_keys, images):
                self.write_image(f, out_key, im)

    def run(self, input_files, output_files, n_jobs=1):
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(self.write_bg_image, input_files, output_files),
                      desc='write background images',
                      total=len(input_files)))
