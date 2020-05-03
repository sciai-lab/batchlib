import os
import numpy as np
from tqdm import tqdm

from .cell_level_analysis import (CellLevelAnalysisBase, CellLevelAnalysisWithTableBase,
                                  compute_ratios)
from ..util.io import open_file, image_name_to_site_name

# default size threshold provided by Vibor
DEFAULT_CELL_OUTLIER_CRITERIA = {'max_size_threshold': 10000,
                                 'min_size_threshold': 1000}

# TODO determine the values for this
DEFAULT_IMAGE_OUTLIER_CRITERIA = {'min_number_cells': None,
                                  'min_number_control_cells': None,
                                  'check_ratios': True}


class CellLevelQC(CellLevelAnalysisBase):
    """ Heuristic quality control for individual cells
    """

    @staticmethod
    def validate_outlier_criteria(outlier_criteria):
        keys = set(outlier_criteria.keys())
        expected_keys = set(DEFAULT_CELL_OUTLIER_CRITERIA.keys())
        if keys != expected_keys:
            raise ValueError("Invalid cell outlier criteria")

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate_bg_median',
                 marker_bg_key='plate_bg_median',
                 table_out_name='outliers',
                 outlier_criteria=DEFAULT_CELL_OUTLIER_CRITERIA,
                 identifier=None,
                 **super_kwargs):
        self.validate_outlier_criteria(outlier_criteria)
        self.outlier_criteria = outlier_criteria

        output_group = cell_seg_key if identifier is None else cell_seg_key + '_' + identifier
        self.table_out_key = output_group + '/' + serum_key + '_' + table_out_name
        super().__init__(cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         output_key='tables/' + self.table_out_key,
                         identifier=identifier,
                         **super_kwargs)

    def cell_level_heuristics(self, cell_stats):
        columns = ['label_id', 'is_outlier', 'outlier_type']

        max_size_threshold = self.outlier_criteria['max_size_threshold']
        min_size_threshold = self.outlier_criteria['min_size_threshold']

        label_ids = cell_stats['labels']
        sizes = cell_stats[self.serum_key]['sizes']

        n_cells = len(label_ids)
        if n_cells != len(sizes):
            raise RuntimeError(f"Labels and sizes are not same length: {n_cells}, {len(sizes)}")

        if max_size_threshold is None:
            outlier_max = np.zeros(n_cells, dtype='bool')
        else:
            outlier_max = sizes > max_size_threshold

        if min_size_threshold is None:
            outlier_min = np.zeros(n_cells, dtype='bool')
        else:
            outlier_min = sizes < min_size_threshold

        is_outlier = np.logical_or(outlier_max, outlier_min).astype('uint8')
        outlier_types = ['too_large' if is_max else ('too_small' if is_min else 'none')
                         for is_min, is_max in zip(outlier_min, outlier_max)]

        table = np.array([label_ids.tolist(),
                          is_outlier.tolist(),
                          outlier_types]).T
        return columns, table

    def outlier_heuristics(self, in_file, out_file):
        cell_stats = self.load_per_cell_statistics(in_file, split_infected_and_control=False)
        columns, table = self.cell_level_heuristics(cell_stats)
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.table_out_key, columns, table)

    def run(self, input_files, output_files):
        for in_file, out_file in tqdm(zip(input_files, output_files),
                                      total=len(input_files),
                                      desc='Cell level quality control'):
            self.outlier_heuristics(in_file, out_file)


class ImageLevelQC(CellLevelAnalysisWithTableBase):
    """ Combining heuristic and manual quality control for images
    """

    @staticmethod
    def validate_outlier_criteria(outlier_criteria):
        keys = set(outlier_criteria.keys())
        expected_keys = set(DEFAULT_IMAGE_OUTLIER_CRITERIA.keys())
        if keys != expected_keys:
            raise ValueError("Invalid image outlier criteria")

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate_bg_median',
                 marker_bg_key='plate_bg_median',
                 table_out_key='images/outliers',
                 outlier_predicate=lambda im: -1,
                 outlier_criteria=DEFAULT_IMAGE_OUTLIER_CRITERIA,
                 **super_kwargs):

        self.validate_outlier_criteria(outlier_criteria)
        self.outlier_criteria = outlier_criteria

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

    def outlier_heuristics(self, in_file):
        infected_stats, control_stats = self.load_per_cell_statistics(in_file,
                                                                      split_infected_and_control=True)

        n_infected = len(infected_stats['labels'])
        n_control = len(control_stats['labels'])
        n_cells = n_infected + n_control

        outlier_type = ''
        is_outlier = 0

        min_n_cells = self.outlier_criteria['min_number_cells']
        if min_n_cells is not None and n_cells < n_cells:
            is_outlier = 1
            outlier_type += 'too few cells;'

        min_control_cells = self.outlier_criteria['min_number_control_cells']
        if min_control_cells is not None and n_control < min_control_cells:
            is_outlier = 1
            outlier_type += 'too few control cells;'

        # check for negative ratios
        if self.outlier_criteria['check_ratios']:
            ratios = compute_ratios(control_stats, infected_stats, serum_key=self.serum_key)
            for name, val in ratios.items():
                if not name.startswith('ratio'):
                    continue
                if val < 0.:
                    is_outlier = 1
                    outlier_type += f'{name} is negative'

        if outlier_type == '':
            outlier_type = 'none'
        else:
            # strip the last ';'
            outlier_type = outlier_type[:-1]
        return is_outlier, outlier_type

    def write_image_outlier_table(self, input_files):
        column_names = ['image_name', 'site_name', 'is_outlier', 'outlier_type']
        table = []

        for in_file in tqdm(input_files, desc="Image level quality control"):
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            site_name = image_name_to_site_name(image_name)

            # check if this image was marked as outlier manually
            manual_outlier = self.outlier_predicate(image_name)
            if manual_outlier not in (-1, 0, 1):
                raise ValueError(f"Invalid value for outlier {manual_outlier}")

            # check if the image is an outlier according to the heuristics
            qc_outlier, qc_outlier_type = self.outlier_heuristics(in_file)
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
