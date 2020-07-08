import os
from concurrent import futures

import numpy as np
from tqdm import tqdm

from ..util import open_file, get_logger
from .cell_level_analysis import (CellLevelAnalysisWithTableBase,
                                  load_cell_outlier_dict, _load_image_outliers)

logger = get_logger('Workflow.BatchJob.DrugScreenAnalysisCellTable')


class DrugScreenAnalysisCellTable(CellLevelAnalysisWithTableBase):
    selected_feature = 'medians'

    def __init__(self, nucleus_seg_key, backgrounds, allow_incompatible_label_ids=True, **super_kwargs):

        self.nucleus_seg_key = nucleus_seg_key
        self.seg_eroded_key = nucleus_seg_key + '_eroded'
        self.seg_dilated_key = nucleus_seg_key + '_dilated'

        self.channel_dict = {self.seg_eroded_key: ['sensor'],
                             self.seg_dilated_key: ['infection marker',
                                                    'infection marker2']}

        self.backgrounds = backgrounds
        self.allow_incompatible_label_ids = allow_incompatible_label_ids
        self.table_out_key = 'cells/default'

        super().__init__(table_out_keys=[self.table_out_key],
                         check_image_outputs=False,
                         cell_seg_key=self.seg_eroded_key,
                         serum_key=self.channel_dict[self.seg_eroded_key][0],
                         marker_key=None,
                         serum_bg_key=None,
                         marker_bg_key=None,
                         output_key=None,
                         validate_cell_classification=False,
                         **super_kwargs)

    def load_image_outliers(self, input_files):
        image_outlier_table = 'images/outliers'
        return _load_image_outliers(self.name, self.table_out_path, image_outlier_table, input_files)

    def write_cell_table(self, input_files, n_jobs):

        # generate the column names
        column_names = ['image_name', 'label_id'] + [channel_key for _, channel_keys in self.channel_dict.items()
                                                     for channel_key in channel_keys]

        cell_outlier_table_name = f'{self.nucleus_seg_key}/outliers'

        def extract_cell_features(in_file):
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]

            table = None
            label_ids = None

            for seg_key, channel_keys in self.channel_dict.items():
                for channel_key in channel_keys:

                    table_key = f'{seg_key}/{channel_key}'
                    with open_file(in_file, 'r') as f:
                        feature_names, features = self.read_table(f, table_key)

                    this_label_ids = features[:, feature_names.index('label_id')]
                    if label_ids is None:
                        label_ids = this_label_ids
                    else:
                        labels_agree = np.array_equal(label_ids, this_label_ids)
                        if not labels_agree:
                            if not self.allow_incompatible_label_ids:
                                raise RuntimeError("Label ids do not agree")
                            mask_a = np.isin(label_ids, this_label_ids)
                            mask_b = np.isin(this_label_ids, label_ids)
                            label_ids = label_ids[mask_a]
                            features = features[mask_b]

                    # get rid of the and background id
                    features = features[label_ids != 0, :]

                    this_features = features[:, feature_names.index(self.selected_feature)][:, None]
                    bg_val = self.backgrounds[channel_key]
                    this_features -= bg_val

                    n_cells = len(this_features)
                    if table is None:
                        table = np.concatenate([np.array([image_name] * n_cells)[:, None],
                                                label_ids[label_ids != 0][:, None],
                                                this_features],
                                               axis=1)
                    else:
                        assert len(table) == len(this_features), f"{len(table)}, {n_cells}"
                        table = np.concatenate([table, this_features], axis=1)

            assert table.shape[1] == len(column_names), f"{table.shape[1]}, {len(column_names)}"

            # filter nuclei that did not pass QC
            outlier_dict = load_cell_outlier_dict(in_file, cell_outlier_table_name, self.name)
            outlier_ids = np.array([cell_id for cell_id, outlier_val in outlier_dict.items()
                                    if outlier_val[0] == 1], dtype='uint32')
            logger.debug(f"{self.name}: {len(outlier_ids)} / {len(table)} cells did not pass QC")

            qc_mask = ~np.isin(table[:, column_names.index('label_id')], outlier_ids)
            table = table[qc_mask]

            return table

        # filter images that did not pass QC
        image_outlier_dict = self.load_image_outliers(input_files)
        n_inputs = len(input_files)
        input_files = [in_file for in_file in input_files
                       if image_outlier_dict.get(in_file, 0) != 1]
        logger.info(f"{self.name}: processing {len(input_files)} / {n_inputs} images")

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            image_tables = list(tqdm(tp.map(extract_cell_features, input_files),
                                     total=len(input_files),
                                     desc='Generate cell tables from features'))

        table = np.concatenate(image_tables, axis=0)
        assert len(column_names) == table.shape[1]

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.table_out_key, column_names, table, force_write=True)

    def run(self, input_files, output_files, n_jobs=1):
        self.write_cell_table(input_files, n_jobs)
