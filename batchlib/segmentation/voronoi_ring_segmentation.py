from skimage.segmentation import watershed  # for now, just use skimage
from scipy import ndimage as ndi
import skimage.morphology as morph
from tqdm.auto import tqdm
import numpy as np

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file


class VoronoiRingSegmentation(BatchJobOnContainer):
    """
    """
    def __init__(self,
                 input_key, output_key,
                 ring_width,
                 input_pattern='*.h5'):
        super().__init__(input_pattern,
                         input_key=input_key, output_key=output_key,
                         input_ndim=2, output_ndim=2)
        self.ring_width = ring_width

    def segment_image(self, in_path):
        with open_file(in_path, 'r') as f:
            input_seg = self.read_input(f, self.input_key)
        input_mask = input_seg > 0
        distance = ndi.distance_transform_edt(input_mask == 0)
        ring_mask = np.invert(morph.dilation(input_mask, morph.disk(self.ring_width)) ^ input_mask)
        voronoi_ring_seg = watershed(distance, input_seg)
        voronoi_ring_seg[ring_mask] = 0
        return voronoi_ring_seg

    def run(self, input_files, output_files):
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            labels = self.segment_image(in_path)
            with open_file(out_path, 'a') as f:
                self.write_result(f, self.output_key, labels)
