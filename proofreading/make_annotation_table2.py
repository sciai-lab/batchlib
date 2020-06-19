import json
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd

from batchlib.util import read_table, image_name_to_well_name
from tqdm import tqdm

manuscript_plates = [
    "20200417_132123_311",
    "20200417_152052_943",
    "20200420_164920_764",
    "20200420_152417_316",
    "plate1_IgM_20200527_125952_707",
    "plate2_IgM_20200527_155923_897",
    "plate5_IgM_20200528_094947_410",
    "plate6_IgM_20200528_111507_585",
    "plate9_4_IgM_20200604_212451_328",
    "plate9_4rep1_20200604_175423_514",
    "plate9_5_IgM_20200605_084742_832",
    "plate9_5rep1_20200604_225512_896"
]


def score_images(plate_folder, old_names):
    plate_name = os.path.split(plate_folder)[1]

    cache_path = f'./plate_stats/{plate_name}.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    images = glob(os.path.join(plate_folder, '*.h5'))
    table_path = os.path.join(plate_folder, f'{plate_name}_table.hdf5')

    with h5py.File(table_path, 'r') as f:
        im_cols, im_tab = read_table(f, 'images/default')
        well_cols, well_tab = read_table(f, 'wells/default')

        im_names = im_tab[:, im_cols.index('image_name')]
        im_outliers = {name: iso for name, iso in zip(im_names,
                                                      im_tab[:, im_cols.index('IgG_is_outlier')])}

        well_names = well_tab[:, well_cols.index('well_name')]
        well_outliers = {name: iso for name, iso in zip(well_names,
                                                        well_tab[:, well_cols.index('IgG_is_outlier')])}

    im_names = []
    ratios = []

    tab_name = 'cell_classification/cell_segmentation/marker_tophat'
    for im in images:
        im_name = os.path.splitext(os.path.split(im)[1])[0]
        # check for outliers
        if im_outliers[im_name] == 1:
            continue
        well_name = image_name_to_well_name(im_name)
        if well_outliers[well_name] == 1:
            continue
        res_name = f'{plate_name}_{im_name}.h5'
        if res_name in old_names:
            continue

        with h5py.File(im, 'r') as f:
            cols, tab = read_table(f, tab_name)

        # compute the infected to non infected ratio
        n_infected = tab[:, cols.index('is_infected')].sum()
        n_control = tab[:, cols.index('is_control')].sum()

        if n_infected == 0 or n_control == 0:
            continue

        ratio = abs(1. - n_infected / float(n_control))
        ratios.append(ratio)
        im_names.append(im_name)

    # sort by the ratio
    im_names = np.array(im_names)
    ratios = np.array(ratios)

    ratio_sorted = np.argsort(ratios)
    im_names = im_names[ratio_sorted].tolist()

    with open(cache_path, 'w') as f:
        json.dump(im_names, f)

    return im_names


def select_images(images, n, out_table):
    im_id = 0
    plate_id = 0

    plate_names = list(images.keys())
    n_plates = len(plate_names)

    for _ in range(n):
        plate_name = plate_names[plate_id]
        im_name = images[plate_name][im_id]
        if im_name.endswith('.h5.h5'):
            im_name = im_name[:-3]
        out_table.append([plate_name, im_name])

        plate_id += 1
        if plate_id == n_plates:
            plate_id = 0
            im_id += 1

    return out_table


def make_second_annotation_table(n_new, n_old):
    root = '/g/kreshuk/data/covid/data-processed'

    old_table = pd.read_excel('./Stacks2proofread.xlsx')
    old_plates = old_table['plate'].values
    old_file_names = old_table['file'].values
    old_names = [f'{plate}_{name}' for plate, name in zip(old_plates, old_file_names)]

    images = {}
    for plate in tqdm(manuscript_plates):
        plate_folder = os.path.join(root, plate)
        ims = score_images(plate_folder, old_names)
        images[plate] = ims

    out_table = []
    out_table = select_images(images, n_new, out_table)

    old_images = {}
    for plate, im_name in zip(old_plates, old_file_names):
        if plate in old_images:
            old_images[plate].append(im_name)
        else:
            old_images[plate] = [im_name]

    out_table = select_images(old_images, n_old, out_table)

    df = pd.DataFrame(out_table, columns=['plate', 'file'])
    df.to_excel('./Stacks2proofread_round2.xlsx', index=False)


if __name__ == '__main__':
    make_second_annotation_table(50, 10)
