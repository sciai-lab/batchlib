import os
from glob import glob

import numpy as np
import pandas as pd
from ..util import read_table, open_file, image_name_to_site_name, image_name_to_well_name

SUPPORTED_TABLE_FORMATS = {'excel': '.xlsx',
                           'csv': '.csv',
                           'tsv': '.tsv'}
# TODO more scores to report?
DEFAULT_SCORE_PATTERNS = ['robust_z_score_of_means',
                          'ratio_of_q0.5_of_means']


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


def export_default_table(table_file, table_name, output_path, output_format=None, skip_existing=True):

    if os.path.exists(output_path) and skip_existing:
        return

    # table file can be a file path or an opened file object
    if isinstance(table_file, str):
        with open_file(table_file, 'r') as f:
            columns, table = read_table(f, table_name)
    else:
        columns, table = read_table(table_file, table_name)
    export_table(columns, table, output_path, output_format)


def export_cell_tables(folder, output_path, table_name, output_format=None, skip_existing=True):
    if os.path.exists(output_path) and skip_existing:
        return

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


def export_voronoi_tables(folder, ext='.xlsx', skip_existing=True):
    plate_name = os.path.split(folder)[1]
    in_file = glob(os.path.join(folder, '*.h5'))[0]
    with open_file(in_file, 'r') as f:
        assert 'tables' in f
        table_names = list(f['tables'].keys())
        voronoi_tables = [name for name in table_names if 'voronoi' in name]
        voronoi_tables = [f'{name}/{channel}' for name in voronoi_tables for channel in f['tables'][name].keys()]

    for table_name in voronoi_tables:
        out_name = f'{plate_name}_cell_table_{table_name}{ext}'.replace('/', '_')
        out_path = os.path.join(folder, out_name)
        export_cell_tables(folder, out_path, table_name, skip_existing=skip_existing)


def export_tables_for_plate(folder,
                            cell_table_name='cell_segmentation',
                            marker_name='marker',
                            skip_existing=True,
                            export_voronoi=True,
                            ext='.xlsx'):
    """ Conveneince function to export all relevant tables for a plate
    into a more common format (by default excel).
    """
    plate_name = os.path.split(folder)[1]
    print("Making summary tables for", plate_name)
    table_file = os.path.join(folder, plate_name + '_table.hdf5')

    # export the images default table
    im_out = os.path.join(folder, f'{plate_name}_image_table{ext}')
    export_default_table(table_file, 'images/default', im_out, skip_existing=skip_existing)

    # export the wells default table
    well_out = os.path.join(folder, f'{plate_name}_well_table{ext}')
    export_default_table(table_file, 'wells/default', well_out, skip_existing=skip_existing)

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
        export_cell_tables(folder, cell_out, cell_table_name, skip_existing=skip_existing)

    # export the infected/non-infected classification
    # cell_table_root = cell_table_name.split('/')[0]
    class_name = f'cell_classification/cell_segmentation/{marker_name}'
    class_out = os.path.join(folder, f'{plate_name}_cell_table_infected_clasification{ext}')
    export_cell_tables(folder, class_out, class_name, skip_existing=skip_existing)

    # export voronoi tables
    if export_voronoi:
        export_voronoi_tables(folder, ext, skip_existing=skip_existing)


def _get_db_metadata(well_names, metadata_repository, plate_name):
    """
    Args
        metadata_repository: instance of PlateMetadataRepository
        plate_name: name of the plate
    Returns:
        additional metadata DB
    """

    def _get_cohort_type(c_id):
        if c_id is None:
            return None
        assert isinstance(c_id, str)
        patient_type = c_id[0].lower()

        if patient_type == 'c':
            return 'positive'
        elif patient_type == 'b':
            return 'control'
        else:
            return 'unknown'

    cohort_ids = metadata_repository.get_cohort_ids(plate_name)
    elisa_results = metadata_repository.get_elisa_results(plate_name)

    additional_values = []
    for well_name in well_names:
        cohort_id = cohort_ids.get(well_name, None)
        cohort_type = _get_cohort_type(cohort_id)
        elisa_IgG, elisa_IgA = elisa_results.get(well_name, (None, None))
        additional_values.append([cohort_id, cohort_id, elisa_IgG, elisa_IgA, cohort_type])

    return np.array(additional_values)


# FIXME all metadata is blank
# FIXME the name matching triggers for unwanted scores,
# we only want IgG and IgM
# FIXME make score names more succinct
def export_scores(folder_list, output_path,
                  score_patterns=DEFAULT_SCORE_PATTERNS,
                  table_name='wells/default',
                  metadata_repository=None):

    def name_matches_score(name):
        return any(pattern in name for pattern in score_patterns)

    # first pass: find all column names that match the pattern
    result_columns = []
    for folder in folder_list:
        plate_name = os.path.split(folder)[1]
        table_path = os.path.join(folder, plate_name + '_table.hdf5')
        if not os.path.exists(table_path):
            raise RuntimeError(f"Did not find a result table @ {table_path}")

        with open_file(table_path, 'r') as f:
            col_names, _ = read_table(f, table_name)
        assert 'well_name' in col_names
        col_names = [col_name for col_name in col_names if name_matches_score(col_name)]
        result_columns.extend(col_names)

    result_columns = ['well_name'] + list(set(result_columns))
    columns = ['plate_name'] + result_columns

    # append cohort_id, elisa results and cohort_type (positive/control/unknow) if we have db
    if metadata_repository is not None:
        db_metadata = ['cohort_id', 'elisa_IgG', 'elisa_IgA', 'cohort_type']
        columns += db_metadata

    # second pass: load the tables
    table = []
    for folder in folder_list:
        plate_name = os.path.split(folder)[1]
        table_path = os.path.join(folder, plate_name + '_table.hdf5')
        with open_file(table_path, 'r') as f:
            this_result_columns, this_result_table = read_table(f, table_name)

        this_len = len(this_result_table)
        plate_col = np.array([plate_name] * this_len)

        col_ids = [this_result_columns.index(name) if name in this_result_columns else -1
                   for name in result_columns[1:]]
        this_table = [np.array([None] * this_len)[:, None] if col_id == -1 else
                      this_result_table[:, col_id:col_id+1] for col_id in col_ids]
        this_table = np.concatenate([plate_col[:, None]] + this_table, axis=1)

        # extend table with the values from DB
        if metadata_repository is not None:
            metadata = _get_db_metadata(this_table[:, 1], metadata_repository, plate_name)
            assert len(metadata) == len(this_table)
            this_table = np.concatenate([this_table, metadata], axis=1)

        table.append(this_table)

    table = np.concatenate(table, axis=0)
    export_table(columns, table, output_path)
