import os
import signal
from glob import glob
from time import sleep

import h5py
import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean, resize

try:
    import z5py
except ImportError:
    z5py = None

H5_EXTS = ['.h5', '.hdf', '.hdf5']
Z5_EXTS = ['.zr', '.zarr', '.n5']


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handle(*self.signal_received)


def is_dataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True
    if z5py is not None and isinstance(obj, z5py.Dataset):
        return True
    return False


def is_group(obj):
    if isinstance(obj, h5py.Group):
        return True
    if z5py is not None and isinstance(obj, z5py.Group):
        return True
    return False


def open_file(path, mode='r', h5_timeout=5, h5_retry=10):
    ext = os.path.splitext(path)[1]

    if ext.lower() in H5_EXTS:
        n_tries = 0
        # this solves some h5 opening errors
        while n_tries < h5_retry:
            try:
                return h5py.File(path, mode=mode)
            except OSError as e:
                sleep(h5_timeout)
                n_tries += 1
                if n_tries >= h5_retry:
                    raise e

    elif ext.lower() in Z5_EXTS:
        if z5py is None:
            raise ValueError("Need z5py to load %s files" % ext)
        return z5py.File(path, mode=mode)

    raise ValueError("Invalid file extensions %s" % ext)


def image_name_to_site_name(image_name):
    parts = image_name.split('_')
    part_a = parts[1].lstrip('Point')
    part_b = parts[2]
    site_name = part_a + '-' + part_b
    return site_name


def get_image_and_site_names(folder, pattern):
    im_names = glob(os.path.join(folder, pattern))
    im_names = [os.path.split(name)[1] for name in im_names]
    im_names = [os.path.splitext(name)[0] for name in im_names]
    site_names = [image_name_to_site_name(name) for name in im_names]
    return im_names, site_names


def write_table(folder, column_dict, column_names, out_path, pattern='*.h5'):

    im_names, site_names = get_image_and_site_names(folder, pattern)
    table = [im_names, site_names]

    n_cols = len(column_names)
    cols = [[column_dict[name][ii] for name in site_names] for ii in range(n_cols)]
    table += cols

    column_names = ['image', 'site-name'] + column_names
    n_cols = len(column_names)
    n_images = len(im_names)

    table = np.array(table).T
    exp_shape = (n_images, n_cols)
    assert table.shape == exp_shape, "%s, %s" % (table.shape, exp_shape)

    table = pd.DataFrame(table, columns=column_names)
    table.to_csv(out_path, sep='\t', index=False)


def sample_shape(shape, scale_factor, add_incomplete_blocks=False):
    if add_incomplete_blocks:
        sampled = tuple(sh // scale_factor + int((sh % scale_factor) != 0)
                        for sh in shape)
    else:
        sampled = tuple(sh // scale_factor for sh in shape)
    sampled = tuple(max(1, sh) for sh in sampled)
    return sampled


def downscale_image(image, scale_factor, use_nearest=None):
    assert isinstance(scale_factor, int), scale_factor
    interpolation_dtypes = (np.uint8, np.int8, np.int16, np.uint16, np.float32, np.float64)
    dtype = image.dtype
    use_nearest = (dtype not in interpolation_dtypes) if use_nearest is None else use_nearest

    out_shape = sample_shape(image.shape, scale_factor)

    if use_nearest:
        downscaled = resize(image, out_shape, order=0, anti_aliasing=False,
                            preserve_range=True).astype(image.dtype)
    else:
        downscaled = downscale_local_mean(image, (scale_factor, scale_factor)).astype(image.dtype)
        # if the shapes disagree, we need to crop righhtmost pixel in the disagreeing axes
        if downscaled.shape != out_shape:
            crop_axes = [dsh != osh for dsh, osh in zip(downscaled.shape, out_shape)]
            crop = tuple(slice(0, sh) if ax else slice(None) for sh, ax in zip(out_shape, crop_axes))
            downscaled = downscaled[crop]

    assert downscaled.shape == out_shape
    return downscaled


def downsample_image_data(path, scale_factors, out_path):
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

    def _copy_and_downscale_dataset(f, ds, name, scale_factors):
        g = f.require_group(name)

        # copy scale 0
        _copy_dataset(g, ds, 'image', False)

        data = ds[:]
        # downsample the other scales
        for scale_factor in scale_factors:
            downscaled = downscale_image(data, scale_factor)
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


def write_image_information(path, image_information=None, well_information=None, overwrite=False):
    # neither image nor well information ? -> do nothing
    if image_information is None and well_information is None:
        return

    with open_file(path, 'a') as f:
        attrs = f.attrs

        def _write_info(info_key, info_val):
            if not isinstance(info_val, str):
                raise ValueError("Expect %s to be str, got %s" % type(info_key, info_val))
            if not overwrite and info_key in attrs:
                info_val = info_val + "; " + attrs[info_key]
            attrs[info_key] = info_val

        if image_information is not None:
            _write_info('ImageInformation', image_information)

        if well_information is not None:
            _write_info('WellInformation', well_information)


def write_viewer_settings(ds, image, color=None, alpha=1., visible=False, skip=None,
                          percentile_min=1, percentile_max=99, scale_factors=None,
                          channel_information=None, use_nearest=None):
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

    if scale_factors is None:
        scale_factors = [1]
    attrs = {'Color': color, 'Visible': visible, 'Skip': skip, 'Alpha': alpha, 'ScaleFactors': scale_factors}

    # if we have additional channel information, add it
    if channel_information is not None:
        attrs.update({'ChannelInformation': channel_information})

    # if we have an actual color and not glasbey, we need to set the contrast limits
    if color in colors:
        # we use percentiles instead of min max to be more robust to outliers
        mi = np.float64(np.percentile(image, percentile_min))
        ma = np.float64(np.percentile(image, percentile_max))
        attrs.update({'ContrastLimits': [mi, ma]})

    ds.attrs.update(attrs)
