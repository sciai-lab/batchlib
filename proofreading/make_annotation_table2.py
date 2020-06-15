import os
from glob import glob

import h5py
import pandas as pd

from batchlib.util import read_table

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
    images = glob(os.path.join(plate_folder, '*.h5'))
    im_list = {}

    plate_name = os.path.split(plate_folder)[1]
    table_path = os.path.join(plate_folder, f'{plate_name}_table.hdf5')

    with h5py.File(table_path, 'r') as f:
        im_cols, im_tab = read_table(f, 'images/default')
        well_cols, well_tab = read_table(f, 'wells/default')

        im_names = im_tab[:, im_cols.index('image_name')]
        print(im_names)

        im_outliers = {name: iso for name, iso in zip(im_names,
                                                      im_tab[:, im_cols.index('IgG_is_outlier')])}

    for im in images:
        im_name = os.path.split(im)
        # check for outliers


def make_second_annotation_table():
    root = '/g/kreshuk/data/covid/data-processed'


if __name__ == '__main__':
    pass
