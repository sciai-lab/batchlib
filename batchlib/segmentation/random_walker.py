from .seeded_segmentation import SeededSegmentation


class RandomWalker(SeededSegmentation):
    """
    """
    def __init__(self, pmap_key, seed_key, output_key, mask_key=None):
        super().__init__(pmap_key, seed_key, output_key, mask_key=mask_key)

    def segment(self, pmap, seeds):
        raise NotImplementedError
