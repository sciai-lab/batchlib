from concurrent import futures
from functools import partial
from tqdm import tqdm

import numpy as np
from skimage.filters import gaussian
from skimage.morphology import dilation, binary_erosion, disk

from ..base import BatchJobOnContainer
from ..util import open_file, normalize


# TODO
# - implement validate function that
# -- checks that segments are mapped to the same id as their seeds
# -- background mask is (approx) mapped to 0
class SeededSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self, pmap_key, seed_key, output_key, mask_key=None, **super_kwargs):

        input_keys = [pmap_key, seed_key]
        if mask_key is not None:
            input_keys.append(mask_key)

        super().__init__(input_key=input_keys, output_key=output_key,
                         input_ndim=2, output_ndim=2, **super_kwargs)

        self.pmap_key = pmap_key
        self.seed_key = seed_key
        self.mask_key = mask_key

    def process_mask_and_seeds(self, seeds, mask,
                               erode_mask, dilate_seeds, ensure_seeds):
        if dilate_seeds > 0:
            seeds = dilation(seeds, disk(dilate_seeds))

        # process with mask
        if mask is None:
            bg_id = None
        # process without mask
        else:
            # ensure that seeds are not masked out
            if ensure_seeds:
                mask[seeds > 0] = 1

            # invert the mask an erode it
            mask = np.logical_not(mask)
            if erode_mask > 0:
                mask = binary_erosion(mask, disk(erode_mask))

            # set the mask to the background id
            bg_id = seeds.max() + 1
            seeds[mask] = bg_id

        return seeds, mask, bg_id

    def process_pmap(self, pmap, invert_pmap, sigma):
        pmap = normalize(pmap)
        if invert_pmap:
            pmap = 1. - pmap
        if sigma > 0:
            pmap = gaussian(pmap, sigma)
        return pmap

    def segment_image(self, in_path, out_path, invert_pmap, sigma,
                      erode_mask, dilate_seeds, ensure_seeds, **kwargs):
        with open_file(in_path, 'r') as f:
            pmap = self.read_image(f, self.pmap_key)
            seeds = self.read_image(f, self.seed_key)
            if self.mask_key is None:
                mask = None
            else:
                mask = self.read_image(f, self.mask_key).astype('bool')

        pmap = self.process_pmap(pmap, invert_pmap, sigma)
        seeds, mask, bg_id = self.process_mask_and_seeds(seeds, mask,
                                                         erode_mask, dilate_seeds, ensure_seeds)

        labels = self.segment(pmap, seeds, **kwargs)

        # set masked area back to zero
        if mask is not None:
            labels[labels == bg_id] = 0

        with open_file(out_path, 'a') as f:
            self.write_image(f, self.output_key, labels)

    def run(self, input_files, output_files, invert_pmap=False, sigma=2.,
            erode_mask=0, dilate_seeds=0, ensure_seeds=True, n_jobs=1, **kwargs):

        _segment = partial(self.segment_image, invert_pmap=invert_pmap, sigma=sigma,
                           erode_mask=erode_mask, dilate_seeds=dilate_seeds,
                           ensure_seeds=ensure_seeds, **kwargs)
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_segment, input_files, output_files), total=len(input_files)))
