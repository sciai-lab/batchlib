import os
import signal
from glob import glob
from time import sleep
from functools import partial

import h5py
import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean, resize

from ..config import get_default_chunks

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


def image_name_to_well_name(image_name):
    parts = image_name.split('_')
    return parts[1].lstrip('Point')


def image_name_to_site_name(image_name):
    parts = image_name.split('_')
    part_a = parts[1].lstrip('Point')
    part_b = parts[2]
    site_name = part_a + '-' + part_b
    return site_name


def in_file_to_image_name(in_file):
    name = os.path.split(in_file)[1]
    name = os.path.splitext(name)[0]
    return name


def in_file_to_plate_name(in_file):
    return os.path.basename(os.path.dirname(in_file))


def get_image_and_site_names(folder, pattern):
    im_names = [in_file_to_image_name(name) for name in glob(os.path.join(folder, pattern))]
    site_names = [image_name_to_site_name(name) for name in im_names]
    return im_names, site_names


def add_site_name_to_image_table(column_names, table):
    assert column_names[0] == 'image_name'
    column_names = list(column_names)
    column_names.insert(1, 'site_name')
    table = np.array(table).tolist()
    for row in table:
        row.insert(1, image_name_to_site_name(row[0]))
    return column_names, np.array(table)


def get_column_dict(column_names, table, column_name):
    assert column_name in column_names, \
        f'Could not find column name "{column_name}" in available column names {column_names}'
    return dict(zip(table[:, 0], table[:, list(column_names).index(column_name)]))


def table_to_row_dict(table):
    return dict(zip(table[:, 0], table[:, 1:]))


def row_dict_to_table(row_dict):
    return np.stack([np.concatenate([[key], row]) for key, row in row_dict.items()], axis=0)


def plate_table_to_well_table(column_names, table, well_names):
    # table is np.ndarray, column_names a list
    assert column_names[0] == 'plate_name', \
        f'Plate tables need to have "plate_name" as their first column, bug got "{column_names[0]}".'
    assert table.shape[0] == 1, f'Plate tables should have one row only, but got {table.shape[0]}.'

    well_column_names = ['well_name'] + column_names[1:]

    plate_name = table[0, 0]
    row_dict = table_to_row_dict(table)
    well_row_dict = {well: row_dict[plate_name] for well in well_names}
    well_table = row_dict_to_table(well_row_dict)
    return well_column_names, well_table


def well_table_to_image_table(column_names, table, image_names):
    # table is np.ndarray, column_names a list
    assert column_names[0] == 'well_name', \
        f'Well tables need to have "well_name" as their first column, bug got "{column_names[0]}".'

    image_column_names = ['image_name'] + column_names[1:]

    row_dict = table_to_row_dict(table)
    image_row_dict = {image_name: row_dict[image_name_to_well_name(image_name)] for image_name in image_names}
    image_table = row_dict_to_table(image_row_dict)
    return add_site_name_to_image_table(image_column_names, image_table)


def is_table_with_specific_first_columns(first_column_names, obj):
    if isinstance(first_column_names, str):
        first_column_names = [first_column_names]
    if not isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj, np.ndarray):
        return False
    try:
        first_columns = obj[0][:len(first_column_names)]
        return first_columns == list(first_column_names)
    except:
        return False


is_image_table = partial(is_table_with_specific_first_columns, ('image_name', 'site_name'))
is_well_table = partial(is_table_with_specific_first_columns, 'well_name')
is_plate_table = partial(is_table_with_specific_first_columns, 'plate_name')


def to_plate_table(obj):
    if is_plate_table(obj):
        return obj
    else:
        assert False, f'not a plate table: {obj}'


def to_well_table(obj, well_names):
    if is_well_table(obj):
        assert set(obj[1][:, 0]) == set(well_names), f'{set(obj[1][:, 0])}, {set(well_names)}'
        return obj
    else:
        return plate_table_to_well_table(*to_plate_table(obj), well_names)


def to_image_table(obj, image_names):
    if is_image_table(obj):
        assert set(obj[1][:, 0]) == set(image_names), f'{set(obj[1][:, 0])}, {set(image_names)}'
        return obj
    else:
        return well_table_to_image_table(*to_well_table(obj, list(map(image_name_to_well_name, image_names))), image_names)


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
                raise ValueError("Expect %s to be str, got %s" % type(info_key), type(info_val))
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


# read/write images
def read_image(f, key, scale=0, channel=None):
    ds = f[key]
    if is_group(ds):
        ds = ds['s%i' % scale]
    assert is_dataset(ds)
    data = ds[:] if channel is None else ds[channel]
    return data


def _write_single_scale(g, out_key, image):
    chunks = get_default_chunks(image)
    ds = g.require_dataset(out_key, shape=image.shape, dtype=image.dtype,
                           compression='gzip', chunks=chunks)
    ds[:] = image
    return ds


def _write_multi_scale(f, out_key, image, use_nearest, scale_factors):
    g = f.require_group(out_key)
    _write_single_scale(g, "s0", image)

    if scale_factors is None:
        return g

    prev_scale_factor = 1
    # note: scale_factors[0] is always 1
    for scale, scale_factor in enumerate(scale_factors[1:], 1):
        rel_scale_factor = int(scale_factor / prev_scale_factor)
        image = downscale_image(image, rel_scale_factor,
                                use_nearest=use_nearest)
        key = "s%i" % scale
        _write_single_scale(g, key, image)
        prev_scale_factor = scale_factor
    return g


def write_image(f, name, image, viewer_settings={}, scale_factors=None):
    # dimensionality is not to
    # -> this is not in image format and we just writ the data
    if image.ndim != 2:
        g = _write_single_scale(f, name, image)
    # otherwise, write in  multi-scale format
    else:
        use_nearest = viewer_settings.get('use_nearest', None)
        g = _write_multi_scale(f, name, image, use_nearest, scale_factors)

    assert isinstance(viewer_settings, dict)
    write_viewer_settings(g, image,
                          scale_factors=scale_factors,
                          **viewer_settings)


def has_image(f, name):
    if name not in f:
        return False
    return 's0' in f[name]


# read/write tables
def read_table(f, name, table_string_type='U100'):
    key = 'tables/%s' % name
    g = f[key]
    ds = g['cells']
    table = ds[:]

    ds = g['columns']
    column_names = [col_name.decode('utf-8') for col_name in ds[:]]

    def _col_dtype(column):
        try:
            column.astype('int')
            return 'int'
        except ValueError:
            pass
        try:
            column.astype('float')
            return 'float'
        except ValueError:
            pass
        return table_string_type

    # find the proper dtypes for the columns and cast
    dtypes = [_col_dtype(col) for col in table.T]
    columns = [col.astype(dtype) for col, dtype in zip(table.T, dtypes)]
    n_rows = table.shape[0]

    table = [[col[row] for col in columns] for row in range(n_rows)]

    # a bit hacky, but we use pandas to handle the mixed dataset
    df = pd.DataFrame(table)
    return column_names, df.values


def write_table(f, name, column_names, table,
                visible=None, force_write=False, table_string_type='S100'):
    if len(column_names) != table.shape[1]:
        raise ValueError(f"Number of columns does not match: {len(column_names)}, {table.shape[1]}")

    # set None to np.nan
    table[np.equal(table, None)] = np.nan

    # make the table datasets. we follow the layout
    # table/cells - contains the data
    # table/columns - containse the column names
    # table/visible - contains which columns are visible in the plate-viewer

    key = 'tables/%s' % name
    g = f.require_group(key)

    def _write_dataset(name, data):
        if name in g:
            shape = g[name].shape
            if shape != data.shape and force_write:
                del g[name]

        ds = g.require_dataset(name, shape=data.shape, dtype=data.dtype,
                               compression='gzip')
        ds[:] = data

    # TODO try varlen string, and if that doesn't work with java,
    # issue a warning if a string is cut
    # cast all values to numpy string
    _write_dataset('cells', table.astype(table_string_type))
    _write_dataset('columns', np.array(column_names, dtype=table_string_type))

    if visible is None:
        visible = np.ones(len(column_names), dtype='uint8')
    _write_dataset('visible', visible)


def has_table(f, name):
    actual_key = 'tables/%s' % name
    if actual_key not in f:
        return False
    g = f[actual_key]
    if not ('cells' in g and 'columns' in g and 'visible' in g):
        return False
    return True
