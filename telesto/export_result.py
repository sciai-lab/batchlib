import argparse
import numpy as np
import pandas as pd
from batchlib.util import open_file, read_table


def export_result(table_path, table_name, prediction_name, save_path):
    with open_file(table_path, 'r') as f:
        col, tab = read_table(f, table_name)

    image_names = tab[:, col.index('image_name')]
    image_names = np.array([name + '.png' for name in image_names])
    print(image_names[:18])

    predictions = tab[:, col.index(prediction_name)]
    print(predictions[:18])

    res = np.concatenate([image_names[:, None],
                          predictions[:, None]],
                         axis=1)
    res = pd.DataFrame(res, columns=['id', 'label_i'])

    res.to_csv(save_path, index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('table_path')
    parser.add_argument('save_path')
    parser.add_argument('--table_name', default='images/default')
    parser.add_argument('--prediction_name', default='prediction')

    args = parser.parse_args()
    export_result(args.table_path, args.table_name, args.prediction_name, args.save_path)
