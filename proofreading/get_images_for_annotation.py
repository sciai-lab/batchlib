import argparse
import os

import h5py
import numpy as np
import pandas as pd
from batchlib.util import read_image, write_image


def export_for_annotation(in_path, out_path):
    keys = ['serum_IgG', 'nuclei', 'marker', 'cell_segmentation']

    with h5py.File(in_path, 'r') as f, h5py.File(out_path, 'a') as f_out:
        for key in keys:
            im = read_image(f, key)
            write_image(f_out, key, im)


def make_image_list(table, root, root_out):
    os.makedirs(root_out, exist_ok=True)
    table = pd.read_excel(table)

    plate_names = table['plate'].values
    file_names = table['file'].values
    if 'infected' in table:
        status = table['infected'].values
    else:
        status = np.zeros(len(table))

    paths = []
    for plate_name, name, stat in zip(plate_names, file_names, status):
        if stat == 1.:
            continue
        plate_name = plate_name.replace('_IgG', '').replace('_IgA', '')
        path = os.path.join(root, plate_name, name)
        if not path.endswith('.h5'):
            path += '.h5'
        paths.append(path)

    paths = list(set(paths))
    paths.sort()

    for in_path in paths:
        assert os.path.exists(in_path), in_path
        folder, file_name = os.path.split(in_path)
        plate_name = os.path.split(folder)[1]
        out_name = f'{plate_name}_{file_name}'
        out_path = os.path.join(root_out, out_name)
        export_for_annotation(in_path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='./Stacks2proofread.xlsx')
    parser.add_argument('--root', default='/g/kreshuk/data/covid/data-processed')
    parser.add_argument('--root_out', default='/g/kreshuk/data/covid/for_annotation/round1')

    args = parser.parse_args()
    make_image_list(args.table, args.root, args.root_out)
