import os
from glob import glob
import argparse
from batchlib.reporting.write_cell_size_masks import WriteCellSizeMasks


def write_mask(folder, n_jobs):
    table_name = 'cell_segmentation/marker'
    seg_key = 'cell_segmentation'
    scale_factors = [1, 2, 4, 8, 16]
    job = WriteCellSizeMasks(table_name=table_name,
                             cell_seg_key=seg_key,
                             scale_factors=scale_factors)
    print("Start for folder", folder)
    job(folder, n_jobs=n_jobs, force_recompute=True)


def write_all_masks():
    folders = glob('/g/kreshuk/data/covid/data-processed/*')
    folders.sort()
    print("Write masks for", len(folders), "plates")

    for folder in folders:
        write_mask(folder, 24)


if __name__ == '__main__':
    write_all_masks()
