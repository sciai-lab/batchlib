import argparse
import pickle
from multiprocessing import cpu_count

import numpy as np
from batchlib.util import open_file, read_table, write_table
from cv import to_features_and_labels


def run_training(table_path, table_name, save_path, prediction_name='prediction'):
    n_jobs = cpu_count()
    with open(save_path, 'rb') as f:
        rf = pickle.load(f)
    rf.set_params(n_jobs=n_jobs)

    with open_file(table_path, 'r') as f:
        cols, tab = read_table(f, table_name)

    x, _ = to_features_and_labels(cols, tab)
    assert x.ndim == 2

    print("Start prediction")
    predictions = rf.predict(x)

    if prediction_name in cols:
        print("Over-writing old prediction result")
        tab[:, cols.index(prediction_name)] = predictions
    else:
        cols = cols + [prediction_name]
        tab = np.concatenate([tab, predictions[:, None]], axis=1)
    with open_file(table_path, 'a') as f:
        write_table(f, table_name, cols, tab, force_write=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('table_path')
    parser.add_argument('save_path')
    parser.add_argument('--table_name', default='images/default')

    args = parser.parse_args()
    run_training(args.table_path, args.table_name, args.save_path)
