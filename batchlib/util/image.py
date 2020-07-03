import numpy as np
from numbers import Number
from scipy.ndimage import convolve
from skimage.segmentation import relabel_sequential

try:
    import numexpr
except ImportError:
    numexpr = None


def standardize(input_, eps=1e-6):
    mean = np.mean(input_)
    std = np.std(input_)
    return (input_ - mean) / np.clip(std, a_min=eps, a_max=None)


def normalize(input_, eps=1e-6):
    input_ = input_.astype(np.float32)
    input_ -= input_.min()
    input_ /= (input_.max() + eps)
    return input_


def barrel_correction(image, divisor, offset):
    if image.shape != divisor.shape:
        raise ValueError(f'Shape mismatch: ({image.shape}, {divisor.shape}) are not all equal')
    if not isinstance(offset, (Number, np.ndarray)):
        raise ValueError("Invalid offset value")
    if isinstance(offset, np.ndarray) and image.shape != offset.shape:
        raise ValueError(f'Shape mismatch: ({image.shape}, {offset.shape}) are not all equal')

    corrected = ((image.astype('float32') - offset) / divisor).astype('float32') + offset
    return corrected


# copied from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
# to avoid any tf import horrors if possible
def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    if numexpr is None:
        x = (x - mi) / (ma - mi + eps)
    else:
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")

    if clip:
        x = np.clip(x, 0, 1)

    return x


def seg_to_edges(segmentation):
    """ Make 2d edges from 2d segmentation
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def _size_filter(segmentation, min_size):
    seg_ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = seg_ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def size_filter(segmentation, min_size, hmap=None, relabel=True):
    segmentation = _size_filter(segmentation, min_size)
    if segmentation is not None:
        # TODO size filter via watershed
        raise NotImplementedError

    if relabel:
        segmentation = relabel_sequential(segmentation)
    return segmentation
