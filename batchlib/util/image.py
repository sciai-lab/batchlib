import numpy as np
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


def barrel_correction(image, barrel_corrector, offset=550):
    if image.shape != barrel_corrector.shape:
        raise ValueError(f'Shape mismatch: {image.shape} != {barrel_corrector.shape}')
    # cast back to uint16 to keep the same datatype
    corrected = ((image - offset) / barrel_corrector).astype(image.dtype)
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
