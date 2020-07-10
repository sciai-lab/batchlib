import argparse
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from batchlib.util import open_file, read_table, write_table
from tqdm import tqdm


def to_features_and_labels(cols, tab):
    have_group_name = 'group_name' in cols
    have_site_name = 'site_name' in cols
    assert have_group_name != have_site_name

    if have_group_name:
        j_left = cols.index('group_name') + 1
    else:
        j_left = cols.index('site_name') + 1

    if 'label' in cols:
        j_right = cols.index('label')
        y = tab[:, j_right].astype('uint8')
    elif 'prediction' in cols:
        j_right = cols.index('prediction')
        y = None
    else:
        j_right = None
        y = None

    x = tab[:, j_left:j_right].astype('float32')
    return x, y


def run_cv(table_path, table_name, n_folds, n_trees, depth, write_result):
    with open_file(table_path, 'r') as f:
        cols, tab = read_table(f, table_name)

    n_jobs = cpu_count()
    skf = StratifiedKFold(n_splits=n_folds)

    x, y = to_features_and_labels(cols, tab)
    assert y is not None
    assert len(x) == len(y)
    assert x.ndim == 2
    assert y.ndim == 1

    predictions = -1 * np.ones(len(x), dtype='int8')

    for train_idx, test_idx in tqdm(skf.split(x, y),
                                    total=n_folds,
                                    desc='Run cross validation'):
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    n_jobs=n_jobs,
                                    max_depth=depth)
        x_train, y_train = x[train_idx], y[train_idx]
        x_test = x[test_idx]
        rf.fit(x_train, y_train)

        pred = rf.predict(x_test)
        predictions[test_idx] = pred

    assert sum(predictions == -1) == 0

    # evaluate the result
    print("Performed", n_folds, "cross-validation with results:")
    print("accuracy:", accuracy_score(y, predictions))
    print("f1-score:", f1_score(y, predictions))
    print("precision:", precision_score(y, predictions))
    print("recall:", recall_score(y, predictions))

    if write_result:
        if 'prediction' in cols:
            print("Over-writing old prediction result")
            tab[:, cols.index('prediction')] = predictions
        else:
            cols = cols + ['prediction']
            tab = np.concatenate([tab, predictions[:, None]], axis=1)
        with open_file(table_path, 'a') as f:
            write_table(f, table_name, cols, tab, force_write=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('table_path')
    parser.add_argument('--table_name', default='images/default')
    parser.add_argument('--n_folds', default=9, type=int)
    parser.add_argument('--n_trees', default=100, type=int)
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--write_result', default=1, type=int)

    args = parser.parse_args()
    run_cv(args.table_path, args.table_name,
           args.n_folds, args.n_trees, args.depth, bool(args.write_result))
