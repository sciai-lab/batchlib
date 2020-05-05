import os
import unittest
from shutil import rmtree

import numpy as np
from batchlib.util.io import open_file, read_table, write_image


class TestCellLevelQC(unittest.TestCase):
    folder = './out'

    def setUp(self):
        os.makedirs(self.folder, exist_ok=True)

        # make the segmentation
        self.seg = [[0, 0, 0, 0, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 2,
                     3, 3, 3, 2, 2,
                     3, 3, 3, 3, 4]]
        self.seg = np.array(self.seg, dtype='uint64')
        self.cell_seg_key = 'seg'

        im_name = 'WellC01_PointC01_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.h5'
        self.path = os.path.join(self.folder, im_name)
        with open_file(self.path, 'a') as f:
            write_image(f, self.cell_seg_key, self.seg)

        # make dummy images for the serum and marker channel
        dummy = np.zeros(self.seg.shape, dtype='float32')
        with open_file(self.path, 'a') as f:
            write_image(f, 'serum', dummy)
        with open_file(self.path, 'a') as f:
            write_image(f, 'marker', dummy)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def run_wf(self, outlier_criteria):
        from batchlib.analysis.cell_level_analysis import InstanceFeatureExtraction, FindInfectedCells
        from batchlib.analysis.cell_analysis_qc import CellLevelQC
        from batchlib.workflow import run_workflow

        job_dict = {
            InstanceFeatureExtraction: {'build': {'cell_seg_key': self.cell_seg_key}},
            FindInfectedCells: {'build': {'cell_seg_key': self.cell_seg_key}},
            CellLevelQC: {'build': {'cell_seg_key': self.cell_seg_key,
                                    'outlier_criteria': outlier_criteria}}
        }
        run_workflow('test', self.folder, job_dict)

    def test_size_thresholds(self):
        min_size = 4
        max_size = 7
        outlier_criteria = {'min_size_threshold': min_size,
                            'max_size_threshold': max_size}
        self.run_wf(outlier_criteria)

        # check the results
        qc_table_name = 'seg/serum_outliers'
        with open_file(self.path, 'r') as f:
            col_names, table = read_table(f, qc_table_name)

        label_ids = table[:, col_names.index('label_id')]
        outlier_values = table[:, col_names.index('is_outlier')]
        outlier_type = table[:, col_names.index('outlier_type')]

        outlier_values = {label: val for label, val in zip(label_ids, outlier_values)}
        outlier_type = {label: val for label, val in zip(label_ids, outlier_type)}

        cell_ids, cell_sizes = np.unique(self.seg, return_counts=True)
        self.assertTrue(np.array_equal(cell_ids, label_ids))

        for label, size in zip(cell_ids, cell_sizes):
            val, otype = outlier_values[label], outlier_type[label]
            if size < min_size:
                self.assertEqual(val, 1)
                self.assertEqual(otype, 'too_small')
            elif size > max_size:
                self.assertEqual(val, 1)
                self.assertEqual(otype, 'too_large')
            else:
                self.assertEqual(val, 0)
                self.assertEqual(otype, 'none')


class TestImageLevelQC(unittest.TestCase):
    folder = './out'

    def get_image_path(self, image_id):
        name_pattern = 'WellC01_PointC0%i_000%i_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.h5'
        return os.path.join(self.folder, name_pattern % (image_id + 1, image_id))

    # TODO write images that: -> we need this for the cell level
    # - violate number of cell criteria
    # - violate number of control cell criteria
    # - violate the ratio criteria
    def setUp(self):
        os.makedirs(self.folder, exist_ok=True)

        # make the segmentations
        self.cell_seg_key = 'seg'
        self.segs = []
        shape = (32, 32)

        # number 1 + 2 + 3: passes QC
        self.segs.append(np.random.randint(0, 100, size=shape, dtype='uint64'))
        self.segs.append(np.random.randint(0, 100, size=shape, dtype='uint64'))
        self.segs.append(np.random.randint(0, 100, size=shape, dtype='uint64'))
        # number 4: fails QC because of too few cells
        self.segs.append(np.ones(shape, dtype='uint64'))
        # number 5: fails QC because of too many cells
        self.segs.append(np.arange(self.segs[0].size, dtype='uint64').reshape(shape))

        self.exp_heuristic_results = {}
        for ii, seg in enumerate(self.segs):
            path = self.get_image_path(ii)
            name = os.path.splitext(os.path.split(path)[1])[0]

            if ii < 3:
                res = (0, 'none')
            elif ii == 3:
                res = (1, 'too few cells')
            elif ii == 4:
                res = (1, 'too many cells')

            self.exp_heuristic_results[name] = res

            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, seg)

            # make dummy images for the serum and marker channel
            dummy = np.zeros(seg.shape, dtype='float32')
            with open_file(path, 'a') as f:
                write_image(f, 'serum', dummy)
            with open_file(path, 'a') as f:
                write_image(f, 'marker', dummy)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def run_wf(self, outlier_criteria):
        from batchlib.analysis.cell_level_analysis import (InstanceFeatureExtraction,
                                                           FindInfectedCells)
        from batchlib.analysis.cell_analysis_qc import ImageLevelQC
        from batchlib.workflow import run_workflow

        job_dict = {
            InstanceFeatureExtraction: {'build': {'cell_seg_key': self.cell_seg_key}},
            FindInfectedCells: {'build': {'cell_seg_key': self.cell_seg_key}},
            ImageLevelQC: {'build': {'cell_seg_key': self.cell_seg_key,
                                     'outlier_criteria': outlier_criteria}}
        }
        run_workflow('test', self.folder, job_dict)

    def test_qc(self):
        outlier_criteria = {'min_number_cells': 10,
                            'max_number_cells': 1000}
        self.run_wf(outlier_criteria)

        path = os.path.join(self.folder, 'out_table.hdf5')
        self.assertTrue(os.path.exists(path))
        qc_table_name = 'images/outliers'
        with open_file(path, 'r') as f:
            col_names, table = read_table(f, qc_table_name)

        n_images = len(self.segs)
        n_cols = 4
        exp_shape = (n_images, n_cols)

        self.assertEqual(len(col_names), n_cols)
        self.assertEqual(table.shape, exp_shape)

        im_name_col = col_names.index('image_name')
        outlier_col = col_names.index('is_outlier')
        outlier_type_col = col_names.index('outlier_type')
        results = {row[im_name_col]: (row[outlier_col], row[outlier_type_col])
                   for row in table}

        keyword = 'heuristic: '
        exp_results = self.exp_heuristic_results
        for im_name, (res_outlier, res_type) in results.items():
            self.assertIn(im_name, exp_results)
            exp_outlier, exp_type = exp_results[im_name]
            self.assertEqual(res_outlier, exp_outlier)

            idx = res_type.find(keyword) + len(keyword)
            res_type = res_type[idx:]
            self.assertEqual(res_type, exp_type)


if __name__ == '__main__':
    unittest.main()
