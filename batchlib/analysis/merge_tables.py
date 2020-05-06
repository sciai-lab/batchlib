import numpy as np

from ..base import BatchJobOnContainer
from ..util import get_logger, open_file


logger = get_logger('Workflow.BatchJob.MergeAnalysisTables')


class MergeAnalysisTables(BatchJobOnContainer):
    """ Merge analysis tables written by CellLevelAnalysis for different parameters.
    """
    image_table_name = 'images/default'
    well_table_name = 'wells/default'

    def __init__(self, input_table_names, reference_table_name, **super_kwargs):

        if reference_table_name not in input_table_names:
            raise ValueError(f"{reference_table_name} was not found in {input_table_names}")

        self.input_table_names = input_table_names
        self.reference_table_name = reference_table_name

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

    # we only keep columns that match the following names:
    # q0.5* for all tables, also outlier and number of cells for the reference table
    # see also https://github.com/hci-unihd/batchlib/issues/91
    def _get_column_mask(self, column_names, is_reference_table):
        # TODO should be parameters with reasonable defaults
        # do we want the robost_z_score ?
        # common_name_patterns = ('q0.5', 'robust_z_score')
        common_name_patterns = ('q0.5',)
        reference_name_patterns = ('score', 'n_cells', 'n_infected', 'n_control', 'fraction_infected',
                                   'is_outlier', 'outlier_type')

        keep_ids = [ii for ii, name in enumerate(column_names)
                    if any(pattern in name for pattern in common_name_patterns)]
        if is_reference_table:
            keep_ids.extend([ii for ii, name in enumerate(column_names)
                             if any(pattern in name for pattern in reference_name_patterns)])

        col_mask = np.zeros(len(column_names), dtype='bool')
        col_mask[keep_ids] = 1
        return col_mask

    def _format_col_name(self, name, prefix):
        # TODO should we do something more fancy, like removing 'serum'?
        return prefix + '_' + name

    def _get_seg_prefix(self, table_name):
        return table_name

    def _merge_tables(self, in_file, out_file, prefix, out_name):

        table = None
        column_names = None

        with open_file(in_file, 'r') as f:
            for table_name in self.input_table_names:
                this_column_names, this_table = self.read_table(f, f"{prefix}/{table_name}")
                seg_prefix = self._get_seg_prefix(table_name)

                is_reference = table_name == self.reference_table_name
                column_mask = self._get_column_mask(this_column_names, is_reference)
                assert len(column_mask) == len(this_column_names)

                this_table = [col for col, is_in_mask in zip(this_table.T, column_mask) if is_in_mask]
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
                    column_names.extend(column_names)

        table = np.array(table)
        assert table.shape[1] == len(column_names), f"{table.shape[1]}, {len(column_names)}"

        with open_file(out_file, 'a') as f:
            self.write_table(f, out_name, column_names, table)

    def merge_image_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'images', self.image_table_name)

    def merge_well_tables(self, in_file, out_file):
        self._merge_tables(in_file, out_file, 'wells', self.well_table_name)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)
        self.merge_well_tables(in_file, out_file)
