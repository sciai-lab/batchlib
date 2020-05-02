import os
import numpy as np
from .cell_level_analysis import CellLevelAnalysisBase, CellLevelAnalysisWithTableBase
from ..util.io import open_file, image_name_to_site_name


class CellLevelQC(CellLevelAnalysisBase):
    """ Heuristic quality control for individual cells
    """

    # TODO output key
    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 table_out_name='outliers'):
        pass

    def run(self, input_files, output_files):
        pass


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
                         output_key=None,
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
            # if it was marked, this image is set to be an outlier
            if manual_outlier == 1:
                outlier_type = 'manual'
                table.append([image_name, site_name, 1, outlier_type])
                continue

            # if the image was not marked as outlier, run the heuristic outlier detection
            qc_outlier, qc_outlier_type = self.outlier_heuristics(image_name)

            if qc_outlier == 0:
                outlier = 0
                outlier_type = 'none'
            elif qc_outlier == 1:
                outlier = 1
                outlier_type = qc_outlier_type
            elif qc_outlier == -1:
                # if we don't have heuristic qc, we take the manual outlier annotation
                outlier = manual_outlier
                outlier_type = 'none' if manual_outlier == 0 else 'not checked'
            else:
                raise RuntimeError(f"Unexpected outlier value: {qc_outlier}")

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
