import os
import pickle
from glob import glob

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from batchlib.analysis.cell_level_analysis import InstanceFeatureExtraction
from batchlib.analysis.cell_analysis_qc import CellLevelQC
from batchlib.util import read_table, write_table, open_file


# TODO
def copy_labels(in_path, out_path):
    in_table_key = ''
    with open_file(in_path, 'r') as f:
        col_names, table = read_table(f, in_table_key)

    with open_file(out_path, 'a') as f:
        write_table(f)


def compute_features_and_labels(gt_folder, out_folder):

    # TODO
    # 0.) compute the plate-wise background and subtract it from the marker channel

    # 1.) compute the features using InstanceFeatureExtraction
    cell_seg_key = ''
    extractor = InstanceFeatureExtraction(channel_keys=['marker_subtracted'],
                                          cell_seg_key=cell_seg_key)
    extractor(out_folder, input_folder=gt_folder)

    # 2.) compute cell outliers
    qc = CellLevelQC(cell_seg_key=cell_seg_key,
                     serum_key='serum_IgG')
    qc(out_folder, input_folder=gt_folder)

    # 3.) copy the labels
    in_paths = glob(os.path.join(gt_folder, '*.h5'))
    for path in in_paths:
        out_path = os.path.join(out_folder, os.path.splt(path)[1])
        copy_labels(path, out_path)


# TODO table names
def load_features_and_labels(path, feature_table_name='', label_table_name='', outlier_table_name=None):
    with open_file(path, 'r') as f:
        cols, table = read_table(f, feature_table_name)

    label_ids = table[:, cols.index('label_ids')]

    marker_pattern = 'marker'
    feat_names = [name for name in cols if marker_pattern in name]
    feat_ids = [ii for ii, name in enumerate(cols) if marker_pattern in name]
    feats = table[:, feat_ids]

    with open_file(path, 'r') as f:
        cols, table = read_table(f, label_table_name)

    this_label_ids = table[:, cols.index('label_ids')]
    assert np.array_equal(label_ids, this_label_ids)
    labels = table[:, cols.index('infected_labels')]

    # filter out outliers from the training data if we have an outlier table
    if outlier_table_name is not None:
        with open_file(path, 'r') as f:
            cols, table = read_table(f, outlier_table_name)
        outlier_col_name = 'is_outlier'
        outlier = table[:, cols.index(outlier_col_name)]
        mask = outlier != 1
    else:
        mask = None

    if mask is not None:
        feats, labels = feats[mask], labels[mask]

    return feats, labels, feat_names


def train_rf(gt_folder, save_path, n_trees=150, max_depth=10, n_threads=4):
    gt_files = glob(os.path.join(gt_folder, '*.h5'))

    feat_names = None
    x, y = [], []
    for path in gt_files:
        feats, labs, this_feat_names = load_features_and_labels(path)

        if feat_names is None:
            feat_names = this_feat_names
        else:
            assert feat_names == this_feat_names

        x.append(feats)
        y.append(labs)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    assert len(x) == len(y)

    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_threads,
                                max_depth=max_depth)
    rf.fit(x, y)
    # monkey patch the feature names, so we can validate this later
    rf.feature_names = feat_names
    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)
