import os
import numpy as np
from .cell_level_analysis import CellLevelAnalysisBase, CellLevelAnalysisWithTableBase
from ..util.io import open_file, image_name_to_site_name


class CellLevelQC(CellLevelAnalysisBase):
    """ Heuristic quality control for individual cells
    """

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate_bg_median',
                 marker_bg_key='plate_bg_median',
                 table_out_name='outliers',
                 **super_kwargs):
        super().__init__(cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         **super_kwargs)
        output_group = cell_seg_key if self.identifier is None else cell_seg_key + '_' + self.identifier
        self.table_out_key = output_group + '/' + serum_key

    # TODO compute actual cell level outliers based on cell features
    def cell_level_heuristics(self, cell_stats):
        n_cells = len(cell_stats)
        columns = ['label_id', 'is_outlier', 'outlier_type']
        table = np.array([list(range(n_cells)),
                          [-1] * n_cells,
                          ['not checked'] * n_cells])
        return columns, table

    def outlier_heuristics(self, in_file, out_file):
        cell_stats = self.load_per_cell_statistics(in_file, split_infected_and_control=False)
        columns, table = self.cell_level_heuristics(cell_stats)
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.table_out_key, columns, table)

    def run(self, input_files, output_files):
        # TODO parallelize and tqdm
        for in_file, out_file in zip(input_files, output_files):
            self.outlier_heuristics(in_file, out_file)


class ImageLevelQC(CellLevelAnalysisWithTableBase):
    """ Combining heuristic and manual quality control for images
    """

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate_bg_median',
                 marker_bg_key='plate_bg_median',
                 table_out_key='images/outliers',
                 outlier_predicate=lambda im: -1,
                 **super_kwargs):

        self.outlier_predicate = outlier_predicate
        self.table_out_key = table_out_key
        super().__init__(table_out_keys=[table_out_key],
                         check_image_outputs=False,
                         cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         **super_kwargs)

    # TODO implement this
    # Ideas:
    # - number of cells
    # - cell size distribution
    # - negative ratios
    def outlier_heuristics(self, image_name):
        return -1, 'not checked'

    def write_image_outlier_table(self, input_files):
        column_names = ['image_name', 'site_name', 'is_outlier', 'outlier_type']
        table = []

        # TODO parallelize and tqdm
        for in_file in input_files:
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            site_name = image_name_to_site_name(image_name)

            # check if this image was marked as outlier manually
            manual_outlier = self.outlier_predicate(image_name)
            if manual_outlier not in (-1, 0, 1):
                raise ValueError(f"Invalid value for outlier {manual_outlier}")

            # check if the image is an outlier according to the heuristics
            qc_outlier, qc_outlier_type = self.outlier_heuristics(image_name)
            if qc_outlier not in (-1, 0, 1):
                raise ValueError(f"Invalid value for outlier {qc_outlier}")

            # check the heuristic outlier status
            outlier_type = f'manual: {manual_outlier}; heuristic: {qc_outlier_type}'
            outlier = qc_outlier

            # if we have a manual outlier or no heuristic check was done,
            # we over-ride the heuristic result
            if (manual_outlier == 1) or (qc_outlier == -1):
                outlier = manual_outlier

            table.append([image_name, site_name, outlier, outlier_type])

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        # set image name to non-visible for the plateViewer
        visible = np.ones(n_cols, dtype='uint8')
        visible[0] = False

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.table_out_key, column_names, table, visible)

    def run(self, input_files, output_files):
        self.write_image_outlier_table(input_files)


class WellLevelQC(CellLevelAnalysisWithTableBase):
    """ Combining heuristic and manual quality control for wells
    """

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 table_out_key='wells/outlier'):
        pass

    def run(self, input_files, output_files):
        pass
