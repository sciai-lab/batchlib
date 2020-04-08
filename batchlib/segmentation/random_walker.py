from .seeded_segmentation import SeededSegmentation


class RandomWalker(SeededSegmentation):
    """
    """
    def __init__(self, pmap_key, seed_key, output_key,
                 mask_key=None, input_pattern='*.h5'):
        super().__init__(pmap_key, seed_key, output_key,
                         mask_key=mask_key, input_ndim=input_pattern)

    def segment(self, pmap, seeds):
        raise NotImplementedError
