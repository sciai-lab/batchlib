import argparse
import pickle
from multiprocessing import cpu_count

from sklearn.ensemble import RandomForestClassifier
from batchlib.util import open_file, read_table
from cv import to_features_and_labels


def run_training(table_path, table_name, save_path, n_trees, depth):
    n_jobs = cpu_count()
    rf = RandomForestClassifier(n_estimators=n_trees,
                                n_jobs=n_jobs,
                                max_depth=depth)
    with open_file(table_path, 'r') as f:
        cols, tab = read_table(f, table_name)

    x, y = to_features_and_labels(cols, tab)
    assert y is not None
    assert len(x) == len(y)
    assert x.ndim == 2
    assert y.ndim == 1

    print("Start training")
    rf.fit(x, y)

    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('table_path')
    parser.add_argument('save_path')
    parser.add_argument('--table_name', default='images/default')
    parser.add_argument('--n_trees', default=100, type=int)
    parser.add_argument('--depth', default=10, type=int)

    args = parser.parse_args()
    run_training(args.table_path, args.table_name, args.save_path,
                 args.n_trees, args.depth)
