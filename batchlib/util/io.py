import os
import numpy as np
from time import sleep

import h5py
try:
    import z5py
except ImportError:
    z5py = None

H5_EXTS = ['.h5', '.hdf', '.hdf5']
Z5_EXTS = ['.zr', '.zarr', '.n5']


def open_file(path, mode='r', h5_timeout=5, h5_retry=10):
    ext = os.path.splitext(path)[1]

    if ext.lower() in H5_EXTS:
        # this solves some h5 opening errors
        n_tries = 0
        while n_tries < h5_retry:
            try:
                return h5py.File(path, mode=mode)
            except OSError:
                sleep(h5_timeout)
                n_tries += 1

    elif ext.lower() in Z5_EXTS:
        if z5py is None:
            raise ValueError("Need z5py to load %s files" % ext)
        return z5py.File(path, mode=mode)

    raise ValueError("Invalid file extensions %s" % ext)


def write_viewer_settings(ds, image, color=None, alpha=1., visible=False, skip=None,
                          percentile_min=1, percentile_max=99):
    # if skip is None, we determine it from the dimensionality
    if skip is None:
        if image.ndim > 2:
            skip = True
        else:
            skip = False

    # we only need to write the skip attribute in this case
    if skip:
        ds.attrs['skip'] = skip
        return

    # if color is None, we determine it from the image dtype
    if color is None:
        dtype = image.dtype
        if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64):
            color = 'White'
        else:
            color = 'Glasbey'

    # check the color value that was passed
    colors = ['Gray', 'Red', 'Green', 'Blue', 'White']
    color_maps = ['Glasbey']
    all_colors = colors + color_maps
    assert color in all_colors

    attrs = {'Color': color, 'Visible': visible, 'Skip': skip, 'Alpha': alpha}
    # if we have an actual color and not glasbey, we need to set the contrast limits
    if color in colors:
        # we use percentiles instead of min max to be more robust to outliers
        mi = np.float64(np.percentile(image, percentile_min))
        ma = np.float64(np.percentile(image, percentile_max))
        attrs.update({'ContrastLimits': [mi, ma]})

    ds.attrs.update(attrs)
