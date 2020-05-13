import os
from glob import glob

import numpy as np
import pandas as pd
from ..util import read_table, open_file, image_name_to_site_name, image_name_to_well_name

SUPPORTED_TABLE_FORMATS = {'excel': '.xlsx',
                           'csv': '.csv',
                           'tsv': '.tsv'}


def format_to_extension(format_):
    if format_ not in SUPPORTED_TABLE_FORMATS:
        supported_formats = list(SUPPORTED_TABLE_FORMATS.keys())
        raise ValueError(f"Format {format_} is not supported, expect one of {supported_formats}")
    return SUPPORTED_TABLE_FORMATS[format_]


def extension_to_format(ext):
    _ext_to_format = {v: k for k, v in SUPPORTED_TABLE_FORMATS.items()}
    if ext not in _ext_to_format:
        supported_exts = list(_ext_to_format.keys())
        raise ValueError(f"Extension {ext} is not supported, expect one of {supported_exts}")
    return _ext_to_format[ext]


def export_table(columns, table, output_path, output_format=None):
    if len(columns) != table.shape[1]:
        raise ValueError(f"Number of columns does not match: {len(columns)}, {table.shape[1]}")

    output_format = extension_to_format(os.path.splitext(output_path)[1])\
        if output_format is None else output_format

    if output_format not in SUPPORTED_TABLE_FORMATS:
        supported_formats = list(SUPPORTED_TABLE_FORMATS.keys())
        raise ValueError(f"Format {output_format} is not supported, expect one of {supported_formats}")

    df = pd.DataFrame(table, columns=columns)
    if output_format == 'excel':
        df.to_excel(output_path, index=False)
    elif output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'tsv':
        df.to_csv(output_path, index=False, sep='\t')


def export_default_table(table_file, table_name, output_path, output_format=None):
    # table file can be a file path or an opened file object
    if isinstance(table_file, str):
        with open_file(table_file, 'r') as f:
            columns, table = read_table(f, table_name)
    else:
        columns, table = read_table(table_file, table_name)
    export_table(columns, table, output_path, output_format)


def export_cell_tables(folder, output_path, table_name, output_format=None):
    files = glob(os.path.join(folder, '*.h5'))
    files.sort()

    plate_name = os.path.split(folder)[1]
    initial_columns = ['plate_name', 'well_name', 'site_name']
    columns = None
    table = []

    for path in files:
        with open_file(path, 'r') as f:
            this_columns, this_table = read_table(f, table_name)
        if columns is None:
            columns = initial_columns + this_columns

        image_name = os.path.splitext(os.path.split(path)[1])[0]
        well_name = image_name_to_well_name(image_name)
        site_name = image_name_to_site_name(image_name)

        plate_col = np.array([plate_name] * len(this_table))
        well_col = np.array([well_name] * len(this_table))
        site_col = np.array([site_name] * len(this_table))
        res_table = np.concatenate([plate_col[:, None], well_col[:, None], site_col[:, None], this_table], axis=1)
        table.append(res_table)

    table = np.concatenate(table, axis=0)
    export_table(columns, table, output_path, output_format)


def export_tables_for_plate(folder, cell_table_name='cell_segmentation', ext='.xlsx'):
    """ Conveneince function to export all relevant tables for a plate
    into a more common format (by default excel).
    """
    plate_name = os.path.split(folder)[1]
    table_file = os.path.join(folder, plate_name + '_table.hdf5')

    # export the images default table
    im_out = os.path.join(folder, f'{plate_name}_image_table{ext}')
    export_default_table(table_file, 'images/default', im_out)

    # export the wells default table
    well_out = os.path.join(folder, f'{plate_name}_well_table{ext}')
    export_default_table(table_file, 'wells/default', well_out)

    # export the cell segmentation tables
    im_file = glob(os.path.join(folder, '*.h5'))[0]
    cell_tables_key = f'tables/{cell_table_name}'
    with open_file(im_file, 'r') as f:
        if cell_tables_key not in f:
            raise RuntimeError(f"Could not find {cell_tables_key} in {im_file}")
        g = f[cell_tables_key]
        cell_table_names = [f'{cell_table_name}/' + name for name in g.keys()]

    for cell_table_name in cell_table_names:
        channel_name = cell_table_name.split('/')[-1]
        cell_out = os.path.join(folder, f'{plate_name}_cell_table_{channel_name}{ext}')
        export_cell_tables(folder, cell_out, cell_table_name)

    # export the infected/non-infected classification
    # cell_table_root = cell_table_name.split('/')[0]
    class_name = f'cell_classification/cell_segmentation/marker'
    class_out = os.path.join(folder, f'{plate_name}_cell_table_infected_clasification{ext}')
    export_cell_tables(folder, class_out, class_name)


def export_scores(folder_list, output_path, table_name='wells/default'):
    columns = None
    table = []

    def name_matches_score(name):
        return 'score' in name and ('robust' not in name)

    for folder in folder_list:
        plate_name = os.path.split(folder)[1]
        table_path = os.path.join(folder, plate_name + '_table.hdf5')
        if not os.path.exists(table_path):
            raise RuntimeError(f"Did not find a result table @ {table_path}")

        with open_file(table_path, 'r') as f:
            col_names, this_table = read_table(f, table_name)

        if columns is None:
            assert 'well_name' in col_names
            columns = ['well_name'] + [col_name for col_name in col_names if name_matches_score(col_name)]
            all_columns = ['plate_name'] + columns

        if len(set(columns) - set(col_names)) > 0:
            raise RuntimeError("Columns don't match")

        col_ids = [col_names.index(col_name) for col_name in columns]
        plate_col = np.array([plate_name] * len(this_table))
        score_table = np.concatenate([plate_col[:, None], this_table[:, col_ids]], axis=1)
        table.append(score_table)

    table = np.concatenate(table, axis=0)
    export_table(all_columns, table, output_path)
