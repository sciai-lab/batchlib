from concurrent import futures

import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from tqdm.auto import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, seg_to_edges, get_logger, in_file_to_image_name

logger = get_logger('Workflow.BatchJob.VoronoiRingSegmentation')


class VoronoiRingSegmentation(BatchJobOnContainer):
    """
    """

    def validate_params(self, ring_width, radius_factor):
        have_width = ring_width is not None
        have_fraction = radius_factor is not None
        if not (have_width != have_fraction):
            raise ValueError("Need either ring_width or radius_factor")

        if have_width:
            logger.info(f"{self.name}: using fixed width {ring_width} for dilation")
        else:
            logger.info(f"{self.name}: using radius fraction {radius_factor} for dilation")

    def __init__(self,
                 input_key, output_key,
                 ring_width=None, radius_factor=None,
                 remove_nucleus=True, **super_kwargs):
        super().__init__(input_key=input_key, input_format='image',
                         output_key=output_key, output_format='image',
                         **super_kwargs)

        self.validate_params(ring_width, radius_factor)
        self.ring_width = ring_width
        self.radius_factor = radius_factor
        self.remove_nucleus = remove_nucleus

    def get_dilation_radius(self, seg, im_path):
        if self.ring_width is not None:
            return self.ring_width
        seg_ids, sizes = np.unique(seg, return_counts=True)

        if seg_ids[0] == 0:
            sizes = sizes[1:]

        median_size = np.median(sizes)

        # logging this to determine the nucleus sizes to estimate the nucleus radius for data
        im_name = in_file_to_image_name(im_path)
        logger.debug(f"{self.name}: median nucleus size for {im_name} is {median_size}")

        # we don't divide by pi here on purpose!
        return int(self.radius_factor * np.sqrt(median_size))

    def segment_image(self, in_path):
        with open_file(in_path, 'r') as f:
            input_seg = self.read_image(f, self.input_key)

        input_mask = input_seg > 0
        if input_mask.sum() == 0:
            logger.warning(f"{self.name}: input segmentation for {in_path}/{self.input_key}")
            return np.zeros_like(input_seg)

        dilation_radius = self.get_dilation_radius(input_seg, in_path)

        distance = ndi.distance_transform_edt(input_mask == 0)
        ring_mask = distance < dilation_radius

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
        if not isinstance(radius, int):
            raise ValueError(f'Raidus {radius} should be int, got {type(radius)}')
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
