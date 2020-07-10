import os
import numpy as np
from tqdm import tqdm

from .cell_level_analysis import (CellLevelAnalysisBase, CellLevelAnalysisWithTableBase,
                                  compute_ratios, load_cell_outlier_dict)
from ..util import open_file, image_name_to_site_name, get_logger, read_table

logger = get_logger('Workflow.BatchJob.CellLevelAnalysis')

# FIXME nucleus and cell seg ids don't always match
# thresholds are final (for the pre-print)
# (I went a bit more conservative then the values proposed by Vibor now)
# for the nucleus size, I have measured the median on some test data and found:
# median: 444.25
# this corresponds to a radius of 12 pixels.
# allowing the radius to vary +- 6 pixels we get the size thresholds here,
# which are very close to vibors suggestiosn (100, 1000)
DEFAULT_CELL_OUTLIER_CRITERIA = {'max_cell_size': 12500,
                                 'min_cell_size': 250,
                                 'min_nucleus_size': None,  # 125
                                 'max_nucleus_size': None}  # 1000


# the cell numbers are final (for the pre-print)
DEFAULT_IMAGE_OUTLIER_CRITERIA = {'max_number_cells': 1000,
                                  'min_number_cells': 10}

DEFAULT_WELL_OUTLIER_CRITERIA = {'min_number_control_cells': 100,  # preprint final
                                 'min_number_infected_cells': None,  # need to determine this
                                 'check_ratios': True,
                                 'min_infected_intensity': None}  # will be set per channel


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
                 nucleus_seg_key='nucleus_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate/backgrounds',
                 marker_bg_key='plate/backgrounds',
                 table_out_name='outliers',
                 outlier_criteria=DEFAULT_CELL_OUTLIER_CRITERIA,
                 identifier=None,
                 **super_kwargs):
        self.validate_outlier_criteria(outlier_criteria)
        self.outlier_criteria = outlier_criteria

        self.nucleus_seg_key = nucleus_seg_key

        output_group = cell_seg_key if identifier is None else cell_seg_key + '_' + identifier

        if serum_key is None:
            self.table_out_key = output_group + '/' + table_out_name
        else:
            self.table_out_key = output_group + '/' + serum_key + '_' + table_out_name

        super().__init__(cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         output_key=self.table_out_key,
                         output_format='table',
                         identifier=identifier,
                         validate_cell_classification=False,
                         **super_kwargs)

    def check_size_thresholds(self, in_file, seg_key, min_size, max_size,
                              skip_none=False):

        if (min_size is None and max_size is None) and skip_none:
            return None, None, None

        with open_file(in_file, 'r') as f:
            seg = self.read_image(f, seg_key)

        label_ids, sizes = np.unique(seg, return_counts=True)
        n_ids = len(label_ids)

        if max_size is None:
            outlier_max = np.zeros(n_ids, dtype='bool')
        else:
            logger.debug(f"{self.name}: max size threshold for {seg_key}: {max_size}")
            outlier_max = sizes > max_size

        if min_size is None:
            outlier_min = np.zeros(n_ids, dtype='bool')
        else:
            logger.debug(f"{self.name}: min size threshold for {seg_key}: {min_size}")
            outlier_min = sizes < min_size

        is_outlier = np.logical_or(outlier_max, outlier_min).astype('uint8')
        outlier_types = ['too_large' if is_max else ('too_small' if is_min else 'none')
                         for is_min, is_max in zip(outlier_min, outlier_max)]

        return label_ids, is_outlier, outlier_types

    def cell_level_heuristics(self, in_file):
        columns = ['label_id', 'is_outlier', 'outlier_type']

        max_cell_size = self.outlier_criteria['max_cell_size']
        min_cell_size = self.outlier_criteria['min_cell_size']

        max_nucleus_size = self.outlier_criteria['max_nucleus_size']
        min_nucleus_size = self.outlier_criteria['min_nucleus_size']

        ids_cells, outliers_cells, types_cells = self.check_size_thresholds(in_file, self.cell_seg_key,
                                                                            min_cell_size, max_cell_size)

        ids_nuclei, outliers_nuclei, types_nuclei = self.check_size_thresholds(in_file, self.nucleus_seg_key,
                                                                               min_nucleus_size, max_nucleus_size,
                                                                               skip_none=True)

        if (ids_nuclei is not None) and not np.array_equal(ids_cells, ids_nuclei):
            raise RuntimeError(f"{self.name}: cell and nucleus ids do not agree")

        if ids_nuclei is None:
            logger.debug(f"{self.name}: Did not compute nucleus size thresholds.")
            is_outlier = outliers_cells
            outlier_types = types_cells
        else:
            is_outlier = np.logical_or(outliers_nuclei, outliers_cells).astype('uint8')
            outlier_types = np.array([f'cell:{ctype},nucleus:{ntype}'
                                      for ctype, ntype in zip(types_cells, types_nuclei)])

        table = np.array([ids_cells.tolist(),
                          is_outlier.tolist(),
                          outlier_types]).T
        return columns, table

    def outlier_heuristics(self, in_file, out_file):
        columns, table = self.cell_level_heuristics(in_file)
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
                         validate_cell_classification=False,
                         **super_kwargs)

    def outlier_heuristics(self, in_file):
        cell_stats = self.load_per_cell_statistics(in_file, split_infected_and_control=False)

        # we subtract 1, otherwise the count would include the background as well
        n_cells = len(cell_stats['labels']) - 1

        outlier_type = ''
        is_outlier = 0

        min_n_cells = self.outlier_criteria['min_number_cells']
        if min_n_cells is not None and n_cells < min_n_cells:
            is_outlier = 1
            outlier_type += 'too few cells;'
            logger.debug(f"{self.name}: {in_file} has too few cells; {n_cells} < {min_n_cells}")

        max_n_cells = self.outlier_criteria['max_number_cells']
        if min_n_cells is not None and n_cells > max_n_cells:
            is_outlier = 1
            outlier_type += 'too many cells;'
            logger.debug(f"{self.name}: {in_file} has too many cells; {n_cells} > {max_n_cells}")

        if outlier_type == '':
            outlier_type = 'none'
        else:
            # strip the last ';'
            outlier_type = outlier_type[:-1]
        return is_outlier, outlier_type

    def write_image_outlier_table(self, input_files):
        column_names = ['image_name', 'site_name', 'is_outlier', 'outlier_type']
        table = []

        if self.outlier_criteria['min_number_cells'] is not None:
            logger.info(f"{self.name}: min number of cells per image: {self.outlier_criteria['min_number_cells']}")

        if self.outlier_criteria['max_number_cells'] is not None:
            logger.info(f"{self.name}: max number of cells per image: {self.outlier_criteria['max_number_cells']}")

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
    """ Heuristic quality control for wells
    """
    min_bg_factor = 3

    @staticmethod
    def validate_outlier_criteria(outlier_criteria):
        keys = set(outlier_criteria.keys())
        expected_keys = set(DEFAULT_WELL_OUTLIER_CRITERIA.keys())
        if keys != expected_keys:
            raise ValueError("Invalid well outlier criteria")

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate/backgrounds',
                 marker_bg_key='plate/backgrounds',
                 table_out_key='wells/outliers',
                 cell_outlier_table_name='outliers',
                 outlier_criteria=DEFAULT_WELL_OUTLIER_CRITERIA,
                 **super_kwargs):

        self.validate_outlier_criteria(outlier_criteria)
        self.outlier_criteria = outlier_criteria

        self.table_out_key = table_out_key
        super().__init__(table_out_keys=[table_out_key],
                         check_image_outputs=False,
                         cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         **super_kwargs)

        output_group = cell_seg_key if self.identifier is None else cell_seg_key + '_' + self.identifier
        self.cell_outlier_table = output_group + '/' + serum_key + '_' + cell_outlier_table_name

    # we need to add this so that outlier cells are ignored when computing the ratios
    def load_cell_outliers(self, input_file):
        return load_cell_outlier_dict(input_file, self.cell_outlier_table, self.name)

    def outlier_heuristics(self, in_files, well_name, min_infected_intensity):
        infected_stats, control_stats = self.load_per_cell_statistics(in_files)
        n_infected = len(infected_stats[self.serum_key]['label_id'])
        n_control = len(control_stats[self.serum_key]['label_id'])

        outlier_type = ''
        is_outlier = 0

        min_control = self.outlier_criteria['min_number_control_cells']
        if min_control is not None and n_control < min_control:
            is_outlier = 1
            outlier_type += 'too few control cells;'

        min_infected = self.outlier_criteria['min_number_infected_cells']
        if min_infected is not None and n_infected < min_infected:
            is_outlier = 1
            outlier_type += 'too few infected cells;'

        # check for negative ratios
        if self.outlier_criteria['check_ratios']:
            ratios = compute_ratios(control_stats, infected_stats,
                                    channel_name_dict=dict(serum=self.serum_key,
                                                           marker=self.marker_key))
            for name, val in ratios.items():
                if not name.startswith('ratio'):
                    continue
                if val < 0. or np.isnan(val):
                    logger.debug(f"{self.name}: {well_name} has negative ratio for {name}")
                    is_outlier = 1
                    outlier_type += f'{name} is negative;'

        if min_infected_intensity is not None:
            infected_intensity = np.median(infected_stats[self.serum_key]['means'])
            if infected_intensity < min_infected_intensity:
                is_outlier = 1
                outlier_type += "too low infected cell intensity;"
                msg = f"{self.name}: {well_name} has too low intensity: {infected_intensity} < {min_infected_intensity}"
                logger.debug(msg)

        if outlier_type == '':
            outlier_type = 'none'
        else:
            # strip the last ';'
            outlier_type = outlier_type[:-1]
        return is_outlier, outlier_type

    def load_min_intensity_from_table(self, table_key):
        with open_file(self.table_out_path, 'r') as f:
            cols, table = read_table(f, table_key)
        serum_key = self.serum_key.split('/')[-1]
        channel_mad_key = f'{serum_key}_mad'
        mad = table[:, cols.index(channel_mad_key)]
        return self.min_bg_factor * mad

    def write_well_outlier_table(self, input_files):
        column_names = ['well_name', 'is_outlier', 'outlier_type']
        input_files_per_well = self.group_images_by_well(input_files)

        min_n_control = self.outlier_criteria['min_number_control_cells']
        if min_n_control is not None:
            logger.info(f"{self.name}: min number of control cells per well: {min_n_control}")

        min_n_infected = self.outlier_criteria['min_number_infected_cells']
        if min_n_infected is not None:
            logger.info(f"{self.name}: min number of infected cells per well: {min_n_infected}")

        if self.outlier_criteria['check_ratios']:
            logger.info(f"{self.name}: checking for negative ratios")

        min_intensity = self.outlier_criteria['min_infected_intensity']
        if isinstance(min_intensity, str):
            logger.info(f"{self.name}: load min intensity from table {min_intensity}")
            min_intensity = self.load_min_intensity_from_table(min_intensity)
        if min_intensity is not None:
            logger.info(f"{self.name}: checking for min serum intensity: {min_intensity}")

        table = []
        for well_name, in_files_for_current_well in tqdm(input_files_per_well.items(),
                                                         desc='Well level quality control'):
            outlier, outlier_type = self.outlier_heuristics(in_files_for_current_well, well_name, min_intensity)
            if outlier not in (-1, 0, 1):
                raise ValueError(f"Invalid value for outlier {outlier}")
            table.append([well_name, outlier, outlier_type])

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.table_out_key, column_names, table)

    def run(self, input_files, output_files):
        self.write_well_outlier_table(input_files)
