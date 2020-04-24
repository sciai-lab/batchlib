# if available use vigra watershed, because it's faster and lifts the gil
# if not available, fall back to skimage
try:
    import vigra

    def watershed(pmap, seeds):
        return vigra.analysis.watershedsNew(pmap.astype('float32'), seeds=seeds.astype('uint32'))[0]

except ImportError:
    from skimage.segmentation import watershed

from .seeded_segmentation import SeededSegmentation


class SeededWatershed(SeededSegmentation):
    """
    """
    def __init__(self, pmap_key, seed_key, output_key, mask_key=None, **super_kwargs):
        super().__init__(pmap_key, seed_key, output_key,
                         mask_key=mask_key, **super_kwargs)

    def segment(self, pmap, seeds):
        return watershed(pmap, seeds)
