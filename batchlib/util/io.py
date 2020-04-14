import os
import numpy as np
from time import sleep
from skimage.transform import downscale_local_mean, rescale

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


def write_table():
    pass


def downsample_image_data(path, scale_factors, out_path=None):
    # TODO support in-place
    assert out_path is not None, "Do not support in-place donwn-sampling for now"
    if any(not isinstance(sf, int) for sf in scale_factors):
        raise ValueError("Expect all down-scaling factors to be integers")

    def _copy_attrs(ds_in, ds_out):
        attrs_in = ds_in.attrs
        attrs_out = ds_out.attrs
        for k, v in attrs_in.items():
            attrs_out[k] = v

    def _copy_dataset(f, ds, name, copy_attrs=True):
        d = ds[:]
        ds_out = f.require_dataset(name, shape=d.shape, dtype=d.dtype,
                                   compression='gzip')
        ds_out[:] = d
        if copy_attrs:
            _copy_attrs(ds, ds_out)

    def _downscale(data, scale_factor):
        interpolation_dtypes = (np.uint8, np.int8, np.int16, np.uint16, np.float32, np.float64)
        if data.dtype in interpolation_dtypes:
            return downscale_local_mean(data, (scale_factor, scale_factor)).astype(data.dtype)
        else:
            return rescale(data, scale_factor, order=0, anti_aliasing=False,
                           preserve_range=True).astype(data.dtype)

    def _copy_and_downscale_dataset(f, ds, name, scale_factors):
        g = f.require_group(name)

        # copy scale 0
        _copy_dataset(g, ds, 'image', False)

        data = ds[:]
        # downsample the other scales
        for scale_factor in scale_factors:
            downscaled = _downscale(data, scale_factor)
            out_name = 'image_scale%ix%i' % (scale_factor, scale_factor)
            ds_out = g.require_dataset(out_name, shape=downscaled.shape,
                                       dtype=downscaled.dtype, compression='gzip')
            ds_out[:] = downscaled

        # copy the viewer settings
        _copy_attrs(ds, g)

    with open_file(path, 'r') as fin, open_file(out_path, 'a') as fout:
        # we don't support data-sets nested in groups for now
        for name in fin:
            ds = fin[name]
            if ds.attrs.get('skip', False):
                _copy_dataset(fout, ds, name)
            else:
                _copy_and_downscale_dataset(fout, ds, name, scale_factors)

        # copy the image level attributes
        _copy_attrs(fin, fout)


def write_image_information(path, image_information=None, well_information=None):
    # neither image nor well information ? -> do nothing
    if image_information is None and well_information is None:
        return

    with open_file(path, 'a') as f:
        if image_information is not None:
            if not isinstance(image_information, str):
                raise ValueError("Expect image_information to be str, got %s" % type(image_information))
            f.attrs['ImageInformation'] = image_information

        if well_information is not None:
            if not isinstance(well_information, str):
                raise ValueError("Expect well_information to be str, got %s" % type(well_information))
            f.attrs['WellInformation'] = well_information


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
