import os
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file
from .cell_level_analysis import CellLevelAnalysisWithTableBase


def mad(values):
    med = np.median(values)
    return np.median(np.abs(values - med))


class MergeImageLevelFeatures(BatchJobOnContainer):
    def __init__(self, input_table_names, initial_table_path=None, **super_kwargs):

        in_table_keys = ['images/' + name for name in input_table_names]

        out_key = 'images/default'
        out_format = 'table'

        self.initial_table_path = initial_table_path

        # we store the global tables with .hdf5 ending to keep them separate from image files
        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=in_table_keys,
                         input_format=['table'] * len(in_table_keys),
                         output_key=out_key,
                         output_format=out_format,
                         identifier=None,
                         **super_kwargs)

    def append_label_column(self, table, column_names):
        image_names = table[:, column_names.index('image_name')]

        initial_table = pd.read_csv(self.initial_table_path, sep='\t')
        im_names = initial_table['id'].values
        labels = initial_table['label_i'].values
        label_dict = dict(zip(im_names, labels))

        label_column = np.array([label_dict[im_name] for im_name in image_names])

        table = np.concatenate([table, label_column[:, None]], axis=1)
        column_names.append('label')

        return table, column_names

    def merge_image_tables(self, in_file, out_file):
        table = None
        column_names = None

        for in_key in self.input_key:
            with open_file(in_file, 'r') as f:
                cols, tab = self.read_table(f, in_key)

            if table is None:
                table = tab
                column_names = cols
                continue

            # go over column names, for matching names, make sure that they match,
            # for new names, append the column
            for col_id, col_name in enumerate(cols):
                col = tab[:, col_id]
                if col_name in column_names:
                    exp_col = table[:, column_names.index(col_name)]
                    assert np.array_equal(col, exp_col)
                else:
                    table = np.concatenate([table, col[:, None]], axis=1)
                    column_names.append(col_name)

        if self.initial_table_path is not None:
            table, column_names = self.append_label_column(table, column_names)

        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_key, column_names, table)

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")
        in_file, out_file = input_files[0], output_files[0]
        self.merge_image_tables(in_file, out_file)


class ImageLevelFeatures(CellLevelAnalysisWithTableBase):
    cell_feature_names = ['sums', 'means', 'medians', 'mads', 'sizes']
    reduction_names = ['mean', 'std',
                       'median', 'mad',
                       'min', 'max',
                       'q10', 'q25', 'q75', 'q90']
    reduction_functions = [np.mean, np.std, np.median, mad, np.min, np.max,
                           partial(np.percentile, q=10), partial(np.percentile, q=25),
                           partial(np.percentile, q=75), partial(np.percentile, q=90)]

    def __init__(self, cell_seg_key, serum_key, initial_table_path=None, **super_kwargs):

        # table keys in the plate-wise *_table.hdf5
        self.image_table_key = f'images/{serum_key}'
        self.initial_table_path = initial_table_path

        super().__init__(table_out_keys=[self.image_table_key],
                         check_image_outputs=False,
                         cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=None,
                         serum_bg_key=None,
                         marker_bg_key=None,
                         output_key=None,
                         identifier=serum_key,
                         validate_cell_classification=False,
                         **super_kwargs)

    def get_group_name_dict(self):
        if self.initial_table_path is None:
            return {}
        tab = pd.read_csv(self.initial_table_path, sep='\t')
        cols = tab.columns.values.tolist()
        tab = tab.values

        image_names = tab[:, cols.index('id')]
        group_names = tab[:, cols.index('group')]
        return dict(zip(image_names, group_names))

    def write_image_table(self, input_files):

        initial_column_names = ['image_name', 'group_name']
        column_names = None
        table = []

        group_name_dict = self.get_group_name_dict()

        for ii, in_file in enumerate(tqdm(input_files, desc='generating image table')):
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]

            with open_file(in_file, 'r') as f:
                cell_feature_names, cell_features = self.read_table(f, self.serum_key)

            # get rid of the and background id
            label_ids = cell_features[:, cell_feature_names.index('label_id')]
            cell_features = cell_features[label_ids != 0, :]

            features = []
            feature_names = []
            for red_name, red_function in zip(self.reduction_names, self.reduction_functions):
                for cell_feat_name in self.cell_feature_names:
                    feat_id = cell_feature_names.index(cell_feat_name)
                    features.append(red_function(cell_features[:, feat_id]))
                    feature_names.append(f'{self.identifier}_{cell_feat_name}_{red_name}')

            if column_names is None:
                column_names = initial_column_names + feature_names

            group_name = group_name_dict.get(image_name, '')
            table.append([image_name, group_name] + features)

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.image_table_key, column_names, table, force_write=True)

    def run(self, input_files, output_files):
        self.write_image_table(input_files)
