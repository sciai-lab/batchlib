from skimage.segmentation import watershed  # for now, just use skimage
from scipy import ndimage as ndi
import skimage.morphology as morph
from tqdm.auto import tqdm
import numpy as np

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, seg_to_edges


class VoronoiRingSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self,
                 input_key, output_key,
                 ring_width,
                 input_pattern='*.h5',
                 disk_not_rings=False,
                 ):
        super().__init__(input_pattern,
                         input_key=input_key, output_key=output_key,
                         input_ndim=2, output_ndim=2)
        self.ring_width = ring_width
        self.disks_not_rings = disk_not_rings

    def segment_image(self, in_path):
        with open_file(in_path, 'r') as f:
            input_seg = self.read_image(f, self.input_key)
        input_mask = input_seg > 0
        assert np.mean(input_mask) > 0
        distance = ndi.distance_transform_edt(input_mask == 0)
        ring_mask = morph.dilation(input_mask, morph.disk(self.ring_width))
        if not self. disks_not_rings:
            ring_mask ^= input_mask  # remove nuclei to get the rings
        voronoi_ring_seg = watershed(distance, input_seg)
        voronoi_ring_seg[np.invert(ring_mask)] = 0
        return voronoi_ring_seg

    def run(self, input_files, output_files):
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files),
                                      desc='Computing voronoi ring segmentations'):
            labels = self.segment_image(in_path)
            with open_file(out_path, 'a') as f:
                self.write_image(f, self.output_key, labels)


class ErodeSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self, radius, bg_label=0, **super_kwargs):
        super().__init__(input_ndim=2, output_ndim=2, **super_kwargs)
        self.radius = radius
        if bg_label != 0:
            raise NotImplementedError

    def process_image(self, in_path, out_path):
        with open_file(in_path, 'r') as f:
            seg = self.read_image(f, self.input_key)
        boundaries_and_bg = seg_to_edges(seg)
        seg[morph.dilation(boundaries_and_bg, morph.disk(self.radius))] = 0
        with open_file(out_path, 'a') as f:
            self.write_image(f, self.output_key, seg)

    def run(self, input_files, output_files):
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files),
                                      desc='computing eroded segmentations'):
            self.process_image(in_path, out_path)
