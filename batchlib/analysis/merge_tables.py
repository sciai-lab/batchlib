from copy import deepcopy
import numpy as np

from ..base import BatchJobOnContainer
from ..util import get_logger, open_file


logger = get_logger('Workflow.BatchJob.MergeAnalysisTables')

# we only keep columns that match the following names:
# - all tables: q0.5, robuse_z_scire
# - reference table:  score, number of cells and outliers
# see also https://github.com/hci-unihd/batchlib/issues/91
DEFAULT_COMMON_NAME_PATTERNS = ('q0.5', 'robust_z_score')
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
                 **super_kwargs):

        if reference_table_name not in input_table_names:
            raise ValueError(f"{reference_table_name} was not found in {input_table_names}")

        self.input_table_names = input_table_names
        self.reference_table_name = reference_table_name

        self.common_name_patterns = common_name_patterns
        self.reference_name_patterns = reference_name_patterns

        in_table_keys = ['images/' + in_key for in_key in input_table_names]
        in_table_keys += ['wells/' + in_key for in_key in input_table_names]

        # we store the global tables with .hdf5 ending to keep them separate from image files
        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=in_table_keys,
                         input_format=['table'] * len(in_table_keys),
                         output_key=[self.image_table_name,
                                     self.well_table_name],
                         output_format=['table', 'table'],
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

    def _format_col_name(self, name, prefix):
        dont_add_prefix_patterns = self.fixed_names + self.reference_name_patterns
        if any(pattern in name for pattern in dont_add_prefix_patterns):
            return name
        else:
            return prefix + '_' + name

    def _get_seg_prefix(self, table_name):
        return table_name.replace('serum_', '')

    def _merge_tables(self, in_file, out_file, prefix, out_name, hide_im_name_column=False):

        table = None
        column_names = None

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

        table = np.array(table)
        assert table.shape[1] == len(column_names), f"{table.shape[1]}, {len(column_names)}"

        # make sure we have the columns marking image / well locations
        if prefix == 'wells' and 'well_name' not in column_names:
            raise RuntimeError("Expected well_name column")
        if prefix == 'images' and 'site_name' not in column_names:
            raise RuntimeError("Expected site_name column")

        visible = np.ones(len(column_names))
        if hide_im_name_column:
            assert 'image_name' in column_names
            im_col_id = column_names.index('image_name')
            visible[im_col_id] = 0

        with open_file(out_file, 'a') as f:
            self.write_table(f, out_name, column_names, table, visible)

    def merge_image_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'images', self.image_table_name, hide_im_name_column=True)

    def merge_well_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'wells', self.well_table_name)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)
        self.merge_well_tables(in_file, out_file)
