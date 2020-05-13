from concurrent import futures

import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from skimage.segmentation import watershed  # for now, just use skimage
from tqdm.auto import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, seg_to_edges


class VoronoiRingSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self,
                 input_key, output_key,
                 ring_width,
                 input_pattern='*.h5',
                 remove_nucleus=True,
                 **super_kwargs):
        super().__init__(input_pattern,
                         input_key=input_key, input_format='image',
                         output_key=output_key, output_format='image',
                         **super_kwargs)
        self.ring_width = ring_width
        self.remove_nucleus = remove_nucleus

    def segment_image(self, in_path):
        with open_file(in_path, 'r') as f:
            input_seg = self.read_image(f, self.input_key)
        input_mask = input_seg > 0
        assert np.mean(input_mask) > 0
        distance = ndi.distance_transform_edt(input_mask == 0)
        ring_mask = morph.dilation(input_mask, morph.disk(self.ring_width))
        if self.remove_nucleus:
            ring_mask ^= input_mask  # remove nuclei to get the rings
        voronoi_ring_seg = watershed(distance, input_seg)
        voronoi_ring_seg[np.invert(ring_mask)] = 0
        return voronoi_ring_seg

    def run(self, input_files, output_files, n_jobs=1):

        def _voronoi(in_path, out_path):
            labels = self.segment_image(in_path)
            with open_file(out_path, 'a') as f:
                self.write_image(f, self.output_key, labels)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_voronoi, input_files, output_files),
                      total=len(input_files),
                      desc='Computing voronoi ring segmentations'))


class ErodeSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self, radius, bg_label=0, **super_kwargs):
        super().__init__(input_ndim=2, output_ndim=2, **super_kwargs)
        assert isinstance(radius, int), f'Raidus {radius} should be int, got {type(radius)}'
        self.radius = radius
        if bg_label != 0:
            raise NotImplementedError

    def process_image(self, in_path, out_path):
        with open_file(in_path, 'r') as f:
            seg = self.read_image(f, self.input_key)
        if self.radius > 0:
            ignore_mask = seg_to_edges(seg)
            if self.radius > 1:
                ignore_mask = morph.dilation(ignore_mask, morph.disk(self.radius - 1))
            seg[ignore_mask] = 0
        with open_file(out_path, 'a') as f:
            self.write_image(f, self.output_key, seg)

    def run(self, input_files, output_files, n_jobs=1):
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(self.process_image, input_files, output_files),
                      total=len(input_files),
                      desc='Computing eroded segmentations'))
