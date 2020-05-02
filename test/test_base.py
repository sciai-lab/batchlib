import os
import unittest
from shutil import rmtree

import imageio
import numpy as np
from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file


class ContainerJob(BatchJobOnContainer):
    key = 'data'

    def rw_image(self, inp, outp, tester):
        im = imageio.imread(inp)
        with open_file(outp, 'a') as f:
            self.write_image(f, self.key, im)
        with open_file(outp, 'r') as f:
            im_out = self.read_image(f, self.key)
        tester.assertTrue(np.allclose(im, im_out))

    def rw_table(self, path, tester):
        with open_file(path, 'r') as f:
            im = self.read_image(f, self.key)
        seg = np.random.randint(0, 100, size=im.shape, dtype='uint32')
        ids = np.unique(seg)
        n_rows = len(ids)
        text = np.array(['lorem ipsum'] * len(ids))
        table = np.array([ids, np.random.rand(n_rows), np.random.rand(n_rows), text]).T

        columns = ['label_id', 'score1', 'score2', 'some_text']
        with open_file(path, 'a') as f:
            self.write_table(f, self.key, columns, table)

        with open_file(path, 'r') as f:
            cols_out, table_out = self.read_table(f, self.key)

        # check that the column names agree
        n_cols = len(columns)
        tester.assertEqual(n_cols, len(cols_out))
        for co, co_out in zip(columns, cols_out):
            tester.assertEqual(co, co_out)

        tester.assertEqual(n_cols, table.shape[1])
        # check that the table values agree
        tester.assertEqual(table.shape, table_out.shape)
        for col_id in range(n_cols):
            col = table[:, col_id]
            col_out = table_out[:, col_id].astype(col.dtype)
            tester.assertTrue(np.array_equal(col, col_out))

    def run(self, input_files, output_files, tester):
        for inp, outp in zip(input_files, output_files):
            self.rw_image(inp, outp, tester)
            self.rw_table(outp, tester)


class TestBase(unittest.TestCase):
    in_folder = os.path.join(os.path.split(__file__)[0], '../data/test_data/test')
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_container_job(self):
        job = ContainerJob(input_pattern='*.tiff',
                           output_ext='.h5')
        job(self.folder, input_folder=self.in_folder, tester=self)


if __name__ == '__main__':
    unittest.main()
