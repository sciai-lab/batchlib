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
DEFAULT_COMMON_NAME_PATTERNS = ('ratio_of_q0.5', 'robust_z_score', 'mad', 'q0.5_of_cell',
                                'ratio_of_mean', 'mean_of_cell')
DEFAULT_REFERENCE_NAME_PATTERNS = ('score', 'n_cells', 'n_infected', 'n_control', 'n_outlier_cells',
                                   'fraction_infected', 'is_outlier', 'outlier_type')


# we need to remove the first occurrence of the feature identifier if
# we get it twice, because of stuff we do to the table key beforehand ...
def modify_column_names(col_names, feature_identifiers):
    def modify_name(name):
        counts = 0
        begins = {}
        for idf in feature_identifiers:
            idf_count = name.count(idf)
            counts += idf_count
            if idf_count > 0:
                begins[idf] = name.find(idf)

        if counts > 1:
            begin = min(begins.values())
            to_remove = [idf for idf, beg in begins.items() if beg == begin]
            assert len(to_remove) == 1
            to_remove = to_remove[0] + '_'
            return name.replace(to_remove, '')
        else:
            return name

    return [modify_name(name) for name in col_names]


class MergeMeanAndSumTables(BatchJobOnContainer):
    image_table_name = 'images/default'
    well_table_name = 'wells/default'
    feature_identifiers = ['mean', 'sum']

    def __init__(self):
        in_pattern = '*.hdf5'
        self.image_in_tables = [self.image_table_name + f'_{idf}' for idf in self.feature_identifiers]
        self.well_in_tables = [self.well_table_name + f'_{idf}' for idf in self.feature_identifiers]

        in_keys = self.image_in_tables + self.well_in_tables
        super().__init__(input_pattern=in_pattern,
                         input_key=in_keys,
                         input_format=len(in_keys) * ['table'],
                         output_key=[self.image_table_name,
                                     self.well_table_name],
                         output_format=['table', 'table'])

    def _merge_tables(self, in_file, out_file, in_tables, out_table_name):
        table = None
        column_names = None

        with open_file(in_file, 'r') as f:
            for idf, table_name in zip(self.feature_identifiers, in_tables):
                this_column_names, this_table = self.read_table(f, table_name)
                this_column_names = modify_column_names(this_column_names, self.feature_identifiers)

                # first feature identifier -> just set the table
                if table is None:
                    table = this_table
                    column_names = this_column_names

                # second feature identifier -> replace the columns corresponding to the feature identifier
                else:
                    if table.shape != this_table.shape:
                        raise RuntimeError(f"Incompatible columns: {table.shape}, {this_table.shape}")
                    # make sure the col names are identical
                    if column_names != this_column_names:
                        diff_colls = [ii for ii, (ca, cb) in enumerate(zip(column_names, this_column_names))
                                      if ca != cb]
                        diff_colls_a = [column_names[ii] for ii in diff_colls]
                        diff_colls_b = [this_column_names[ii] for ii in diff_colls]
                        raise RuntimeError(f"Incompatible columns:\n{diff_colls_a}\n{diff_colls_b}")

                    # replace column ids for this feature identifier
                    replace_col_ids = [ii for ii, name in enumerate(column_names) if idf in name]
                    if len(replace_col_ids) == 0:
                        raise RuntimeError(f"Did not find any columns to replace for identifier {idf}")

                    table[:, replace_col_ids] = this_table[:, replace_col_ids]

        visible = np.ones(len(column_names), dtype='uint8')
        if out_table_name == self.image_table_name:
            visible[0] = 0

        with open_file(out_file, 'a') as f:
            self.write_table(f, out_table_name, column_names, table, visible)
        logger.info(f"{self.name}: write merged table to {out_file}:{out_table_name}")

    def merge_image_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, self.image_in_tables, self.image_table_name)

    def merge_well_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, self.well_in_tables, self.well_table_name)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)
        self.merge_well_tables(in_file, out_file)


class MergeAnalysisTables(BatchJobOnContainer):
    """ Merge analysis tables written by CellLevelAnalysis for different parameters.
    """
    fixed_names = ('well_name', 'image_name', 'site_name')

    def __init__(self, input_table_names, reference_table_name,
                 common_name_patterns=DEFAULT_COMMON_NAME_PATTERNS,
                 reference_name_patterns=DEFAULT_REFERENCE_NAME_PATTERNS,
                 analysis_parameters=None, background_column_patterns=('median', 'mad'),
                 identifier=None, hide_sums=True, **super_kwargs):

        if reference_table_name not in input_table_names:
            raise ValueError(f"{reference_table_name} was not found in {input_table_names}")

        self.input_table_names = input_table_names
        self.reference_table_name = reference_table_name

        self.common_name_patterns = common_name_patterns
        self.reference_name_patterns = reference_name_patterns

        self.background_column_patterns = background_column_patterns
        self.image_background_table = 'images/backgrounds'
        self.well_background_table = 'wells/backgrounds'

        self.hide_sums = hide_sums

        in_table_keys = ['images/' + in_key for in_key in input_table_names]
        in_table_keys += ['wells/' + in_key for in_key in input_table_names]
        in_table_keys += [self.image_background_table, self.well_background_table]

        self.image_table_name = 'images/default'
        self.well_table_name = 'wells/default'
        if identifier is not None:
            self.image_table_name += f'_{identifier}'
            self.well_table_name += f'_{identifier}'

        out_keys = [self.image_table_name, self.well_table_name]
        out_format = ['table', 'table']
        if analysis_parameters is None:
            self.analysis_parameters = None
        else:
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
                         identifier=None,
                         **super_kwargs)

    def _get_column_mask(self, column_names, is_reference_table, keep_names, has_marker_values):

        keep_patterns = deepcopy(self.common_name_patterns)
        if keep_names:
            keep_patterns += self.fixed_names
        if is_reference_table:
            keep_patterns += self.reference_name_patterns

        keep_ids = [ii for ii, name in enumerate(column_names)
                    if (any(pattern in name for pattern in keep_patterns) and 'marker' not in name)]

        if not has_marker_values:
            marker_ids = [ii for ii, name in enumerate(column_names)
                          if (any(pattern in name for pattern in keep_patterns) and 'marker' in name)]
            if len(marker_ids) > 0:
                has_marker_values = True
                keep_ids.extend(marker_ids)

        col_mask = np.zeros(len(column_names), dtype='bool')
        col_mask[keep_ids] = 1
        return col_mask, has_marker_values

    def _format_col_name(self, name, prefix):
        is_marker = 'marker' in name
        is_zscore = 'robust_z_score' in name
        override_ = is_zscore and not is_marker

        dont_add_prefix_patterns = self.fixed_names + self.reference_name_patterns + ('marker',)
        if any(pattern in name for pattern in dont_add_prefix_patterns) and not override_:
            return name.replace('serum', '').replace('of_cell_', '')
        else:
            return prefix + name.replace('serum', '').replace('of_cell_', '')

    def _get_table_prefix(self, table_name):
        return table_name.replace('serum_', '')

    def _merge_tables(self, in_file, out_file, prefix, out_name, is_image):

        table = None
        column_names = None
        has_marker_values = False

        # FIXME we assume here that all tables store the reference column (i.e. well_name or site_name)
        # in the same order. If this is not the case, the values will get mixed. We need to check for this!
        with open_file(in_file, 'r') as f:
            for ii, table_name in enumerate(self.input_table_names):
                this_column_names, this_table = self.read_table(f, f"{prefix}/{table_name}")
                table_prefix = self._get_table_prefix(table_name)

                is_reference = table_name == self.reference_table_name
                keep_names = ii == 0
                column_mask, has_marker_values = self._get_column_mask(this_column_names, is_reference,
                                                                       keep_names, has_marker_values)
                assert len(column_mask) == len(this_column_names)

                this_table = [col for col, is_in_mask in zip(this_table.T, column_mask)
                              if is_in_mask]
                this_table = np.array(this_table).T
                this_column_names = [self._format_col_name(col_name, table_prefix)
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

            bg_col_names = [name for name in bg_columns
                            if any(pattern in name for pattern in self.background_column_patterns)]

            logger.info(f'{self.name}: add background columns {bg_col_names}')
            bg_col_ids = [bg_columns.index(name) for name in bg_col_names]
            bg_values = bg_table[:, bg_col_ids]

            table = [row + bg_vals.tolist() for row, bg_vals in zip(table, bg_values)]
            bg_col_names = ['background_' + name.replace('serum_', '') for name in bg_col_names]
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

        # hide the sum columns if specified
        if self.hide_sums:
            sum_cols = [ii for ii, name in enumerate(column_names) if 'sum' in name]
            logger.info(f"{self.name}: hide {len(sum_cols)} columns that contain sum based scores")
            visible[sum_cols] = 0

        logger.info(f"{self.name}: write merged table with columns {column_names}")

        with open_file(out_file, 'a') as f:
            self.write_table(f, out_name, column_names, table, visible, force_write=True)

    def merge_image_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'images', self.image_table_name, is_image=True)

    def merge_well_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'wells', self.well_table_name, is_image=False)

    def write_parameter_table(self, out_file):
        col_names = ['plate_name'] + list(self.analysis_parameters.keys())

        plate_name = os.path.split(out_file)[1]
        table = np.array([plate_name] + list(self.analysis_parameters.values()))[None]
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.parameter_table_name, col_names, table, force_write=True)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)
        self.merge_well_tables(in_file, out_file)
        if self.analysis_parameters is not None:
            self.write_parameter_table(out_file)
