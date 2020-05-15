from concurrent import futures
from functools import partial

import numpy as np
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file


class WriteCellSizeMasks(BatchJobOnContainer):
    def __init__(self, table_name, cell_seg_key='segmentation',
                 min_threshold=100, max_threshold=25000, **super_kwargs):
        self.table_name = table_name
        self.output_key = 'cell_size_mask'
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cell_seg_key = cell_seg_key
        super().__init__(input_key=[self.table_name, self.cell_seg_key],
                         input_format=['table', 'image'],
                         output_key=self.output_key, output_format='image',
                         **super_kwargs)

    def write_size_mask(self, in_file, out_file):
        with open_file(in_file, 'r') as f:
            seg = self.read_image(f, self.cell_seg_key)
            col_names, table = self.read_table(f, self.table_name)

        label_ids = table[:, col_names.index('label_id')]
        sizes = table[:, col_names.index('sizes')]

        seg_ids, counts = np.unique(seg, return_counts=True)
        # We also add a test to check that the size computation was correct
        if not np.array_equal(label_ids, seg_ids):
            raise RuntimeError(f"Label ids dont match for image {in_file}")

        if not np.allclose(counts, sizes):
            raise RuntimeError(f"Cell sizes dont match for image {in_file}")

        small = label_ids[sizes < self.min_threshold]
        large = label_ids[sizes > self.max_threshold]

        size_mask = np.zeros_like(seg)
        size_mask[np.isin(seg, small)] = 1
        size_mask[np.isin(seg, large)] = 1

        with open_file(out_file, 'a') as f:
            self.write_image(f, self.output_key, size_mask)

    def run(self, input_files, output_files, n_jobs=1):
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(self.write_size_mask, input_files, output_files),
                      desc='write size masks',
                      total=len(input_files)))
