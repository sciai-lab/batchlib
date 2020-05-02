from .cell_level_analysis import CellLevelAnalysisBase, CellLevelAnalysisWithTableBase


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
                 table_out_path='images/outliers',
                 outlier_predicate=lambda im: -1):

        self.outlier_predicate = outlier_predicate

    def run(self, input_files, output_files):
        pass


class WellLevelQC(CellLevelAnalysisWithTableBase):
    """ Combining heuristic and manual quality control for wells
    """

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 table_out_path='wells/outlier'):
        pass

    def run(self, input_files, output_files):
        pass
