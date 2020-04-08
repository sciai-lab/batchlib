from concurrent import futures
from functools import partial

import imageio
import numpy as np
from tqdm import tqdm

from ..base import BatchJob
from ..util import barrel_correction, open_file


# TODO consider changing default to n5
class Preprocess(BatchJob):
    def __init__(self, output_ext='.h5'):
        super().__init__(input_pattern='*.tiff', output_ext=output_ext,
                         output_key='raw', output_ndim=3)
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
            assert im.shape[0] == 3
            im = self._reorder(im)
        else:
            # get rid of garbage channels
            assert im.shape[0] == 4
            im = im[:3]

        # apply barrel correction
        if barrel_corrector is not None:
            im = barrel_correction(im, barrel_corrector)

        with open_file(out_path, 'a') as f:
            ds = f.require_dataset(self.output_key, shape=im.shape, dtype=im.dtype,
                                   compression='gzip')
            ds[:] = im

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
