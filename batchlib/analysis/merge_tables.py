import os
from copy import deepcopy
import numpy as np

from ..base import BatchJobOnContainer
from ..util import get_logger, open_file


logger = get_logger('Workflow.BatchJob.MergeAnalysisTables')

# we only keep columns that match the following names:
# - all tables: q0.5, robuse_z_score
# - reference table:  score, number of cells and outliers
# see also https://github.com/hci-unihd/batchlib/issues/91
DEFAULT_COMMON_NAME_PATTERNS = ('q0.5', 'robust_z_score', 'mad')
DEFAULT_REFERENCE_NAME_PATTERNS = ('score', 'n_cells', 'n_infected', 'n_control',
                                   'fraction_infected', 'is_outlier', 'outlier_type')


class MergeAnalysisTables(BatchJobOnContainer):
    """ Merge analysis tables written by CellLevelAnalysis for different parameters.
    """
    image_table_name = 'images/default'
    well_table_name = 'wells/default'
    fixed_names = ('well_name', 'image_name', 'site_name')

    def __init__(self, input_table_names, reference_table_name,
                 common_name_patterns=DEFAULT_COMMON_NAME_PATTERNS,
                 reference_name_patterns=DEFAULT_REFERENCE_NAME_PATTERNS,
                 analysis_parameters=None, background_column_pattern='median',
                 **super_kwargs):

        if reference_table_name not in input_table_names:
            raise ValueError(f"{reference_table_name} was not found in {input_table_names}")

        self.input_table_names = input_table_names
        self.reference_table_name = reference_table_name

        self.common_name_patterns = common_name_patterns
        self.reference_name_patterns = reference_name_patterns

        self.background_column_pattern = background_column_pattern
        self.image_background_table = 'images/backgrounds'
        self.well_background_table = 'wells/backgrounds'

        in_table_keys = ['images/' + in_key for in_key in input_table_names]
        in_table_keys += ['wells/' + in_key for in_key in input_table_names]
        in_table_keys += [self.image_background_table, self.well_background_table]

        out_keys = [self.image_table_name, self.well_table_name]
        out_format = ['table', 'table']
        if analysis_parameters is not None:
            self.analysis_parameters = analysis_parameters
            self.parameter_table_name = 'plate/analysis_parameter'
            out_keys.append(self.parameter_table_name)
            out_format.append('table')

        # we store the global tables with .hdf5 ending to keep them separate from image files
        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=in_table_keys,
                         input_format=['table'] * len(in_table_keys),
                         output_key=out_keys,
                         output_format=out_format,
                         **super_kwargs)

    def _get_column_mask(self, column_names, is_reference_table, keep_names):

        keep_patterns = deepcopy(self.common_name_patterns)
        if keep_names:
            keep_patterns += self.fixed_names
        if is_reference_table:
            keep_patterns += self.reference_name_patterns

        keep_ids = [ii for ii, name in enumerate(column_names)
                    if any(pattern in name for pattern in keep_patterns)]

        col_mask = np.zeros(len(column_names), dtype='bool')
        col_mask[keep_ids] = 1
        return col_mask

    def _format_col_name(self, name, prefix, dont_add_prefix_patterns=None):
        dont_add_prefix_patterns = self.fixed_names + self.reference_name_patterns\
            if dont_add_prefix_patterns is None else dont_add_prefix_patterns
        if any(pattern in name for pattern in dont_add_prefix_patterns):
            return name
        else:
            return prefix + '_' + name

    def _get_seg_prefix(self, table_name):
        return table_name.replace('serum_', '')

    def _merge_tables(self, in_file, out_file, prefix, out_name, is_image):

        table = None
        column_names = None

        # FIXME we assume here that all tables store the reference column (i.e. well_name or site_name)
        # in the same order. If this is not the case, the values will get mixed. We need to check for this!
        with open_file(in_file, 'r') as f:
            for ii, table_name in enumerate(self.input_table_names):
                this_column_names, this_table = self.read_table(f, f"{prefix}/{table_name}")
                seg_prefix = self._get_seg_prefix(table_name)

                is_reference = table_name == self.reference_table_name
                keep_names = ii == 0
                column_mask = self._get_column_mask(this_column_names, is_reference, keep_names)
                assert len(column_mask) == len(this_column_names)

                this_table = [col for col, is_in_mask in zip(this_table.T, column_mask)
                              if is_in_mask]
                this_table = np.array(this_table).T
                this_column_names = [self._format_col_name(col_name, seg_prefix)
                                     for col_name, keep_col in zip(this_column_names,
                                                                   column_mask) if keep_col]

                if table is None:
                    table = [row.tolist() for row in this_table]
                    column_names = this_column_names
                else:
                    if len(table) != len(this_table):
                        raise RuntimeError(f"Invalid number of rows {len(table)}, {len(this_table)}")
                    table = [row + this_row.tolist() for row, this_row in zip(table, this_table)]
                    column_names.extend(this_column_names)
                    assert len(table[0]) == len(column_names)

            # add the relevant background value to the summary table
            bg_table_name = self.image_background_table if is_image else self.well_background_table
            bg_columns, bg_table = self.read_table(f, bg_table_name)
            if len(table) != len(bg_table):
                raise RuntimeError(f"Invalid number of rows {len(table)}, {len(bg_table)}")

            bg_col_names = [name for name in bg_columns if self.background_column_pattern in name]
            bg_col_ids = [bg_columns.index(name) for name in bg_col_names]
            bg_values = bg_table[:, bg_col_ids]

            table = [row + bg_vals.tolist() for row, bg_vals in zip(table, bg_values)]
            column_names.extend(bg_col_names)

        table = np.array(table)
        assert table.shape[1] == len(column_names), f"{table.shape[1]}, {len(column_names)}"

        # make sure we have the columns marking image / well locations
        if (not is_image) and 'well_name' not in column_names:
            raise RuntimeError("Expected well_name column")
        if is_image and 'site_name' not in column_names:
            raise RuntimeError("Expected site_name column")

        visible = np.ones(len(column_names))
        if is_image:
            assert 'image_name' in column_names
            im_col_id = column_names.index('image_name')
            visible[im_col_id] = 0

        with open_file(out_file, 'a') as f:
            self.write_table(f, out_name, column_names, table, visible)

    def merge_image_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'images', self.image_table_name, is_image=True)

    def merge_well_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'wells', self.well_table_name, is_image=False)

    def write_parameter_table(self, out_file):
        col_names = ['plate_name'] + list(self.analysis_parameters.keys())

        plate_name = os.path.split(out_file)[1]
        table = np.array([plate_name] + list(self.analysis_parameters.values()))[None]
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.parameter_table_name, col_names, table)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)
        self.merge_well_tables(in_file, out_file)
        if self.analysis_parameters is not None:
            self.write_parameter_table(out_file)
