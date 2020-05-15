import os
import unittest
from glob import glob
from shutil import rmtree

import numpy as np
import pandas as pd
from batchlib.util.io import open_file, read_image, read_table, write_image


class TestCellLevelQC(unittest.TestCase):
    input_folder = os.path.join(os.path.split(__file__)[0], '../../data/test_data/test')
    misc_folder = os.path.join(os.path.split(__file__)[0], '../../misc')
    n_cpus = 4
    folder = './out'

    def make_toy_data(self):
        os.makedirs(self.folder, exist_ok=True)

        # make the segmentation
        self.seg = [[0, 0, 0, 0, 0,
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

    def run_wf1(self, outlier_criteria):
        from batchlib.analysis.cell_level_analysis import (InstanceFeatureExtraction,
                                                           FindInfectedCells,
                                                           ExtractBackground)
        from batchlib.analysis.cell_analysis_qc import CellLevelQC
        from batchlib.workflow import run_workflow

        job_dict = {
            InstanceFeatureExtraction: {'build': {'cell_seg_key': self.cell_seg_key}},
            ExtractBackground: {'build': {'cell_seg_key': self.cell_seg_key,
                                          'serum_key': 'serum',
                                          'actual_channels_to_use': ['serum', 'marker'],
                                          'marker_key': 'marker'}},
            FindInfectedCells: {'build': {'cell_seg_key': self.cell_seg_key}},
            CellLevelQC: {'build': {'cell_seg_key': self.cell_seg_key,
                                    'outlier_criteria': outlier_criteria}}
        }
        run_workflow('test', self.folder, job_dict)

    def check_size_thresholds(self, path, seg_key, qc_table_name, min_size, max_size):
        with open_file(path, 'r') as f:
            seg = read_image(f, seg_key)
            col_names, table = read_table(f, qc_table_name)

        label_ids = table[:, col_names.index('label_id')]
        outlier_values = table[:, col_names.index('is_outlier')]
        outlier_type = table[:, col_names.index('outlier_type')]

        outlier_values = {label: val for label, val in zip(label_ids, outlier_values)}
        outlier_type = {label: val for label, val in zip(label_ids, outlier_type)}

        cell_ids, cell_sizes = np.unique(seg, return_counts=True)
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

    def test_size_thresholds_toy(self):
        self.make_toy_data()

        min_size = 4
        max_size = 7
        outlier_criteria = {'min_cell_size': min_size,
                            'max_cell_size': max_size,
                            'min_nucleus_size': None,
                            'max_nucleus_size': None}
        self.run_wf1(outlier_criteria)

        # check the results
        qc_table_name = 'seg/serum_outliers'
        self.check_size_thresholds(self.path, self.cell_seg_key, qc_table_name,
                                   min_size, max_size)

    def run_wf2(self, outlier_criteria):
        from batchlib.preprocessing import Preprocess
        from batchlib.analysis.cell_level_analysis import (InstanceFeatureExtraction,
                                                           FindInfectedCells,
                                                           ExtractBackground)
        from batchlib.analysis.cell_analysis_qc import CellLevelQC
        from batchlib.segmentation import SeededWatershed
        from batchlib.segmentation.stardist_prediction import StardistPrediction
        from batchlib.segmentation.torch_prediction import TorchPrediction
        from batchlib.segmentation.unet import UNet2D
        from batchlib.workflow import run_workflow

        model_root = os.path.join(self.misc_folder, 'models/stardist')
        model_name = '2D_dsb2018'

        torch_model_path = os.path.join(self.misc_folder, 'models/torch/fg_and_boundaries_V1.torch')
        torch_model_class = UNet2D
        torch_model_kwargs = {
            'in_channels': 1,
            'out_channels': 2,
            'f_maps': [32, 64, 128, 256, 512],
            'testing': True
        }

        job_dict = {
            Preprocess.from_folder: {'build': {'input_folder': self.input_folder},
                                     'run': {'n_jobs': self.n_cpus}},
            TorchPrediction: {'build': {'input_key': 'serum_IgG',
                                        'output_key': ['mask', 'boundaries'],
                                        'model_path': torch_model_path,
                                        'model_class': torch_model_class,
                                        'model_kwargs': torch_model_kwargs},
                              'run': {'gpu_id': None,
                                      'batch_size': 1,
                                      'threshold_channels': {0: 0.5}}},
            StardistPrediction: {'build': {'model_root': model_root,
                                           'model_name': model_name,
                                           'input_key': 'nuclei',
                                           'output_key': 'nucleus_segmentation'},
                                 'run': {'gpu_id': None,
                                         'n_jobs': self.n_cpus}},
            SeededWatershed: {'build': {'pmap_key': 'boundaries',
                                        'seed_key': 'nucleus_segmentation',
                                        'output_key': 'segmentation',
                                        'mask_key': 'mask'},
                              'run': {'erode_mask': 20,
                                      'dilate_seeds': 3,
                                      'n_jobs': self.n_cpus}},
            InstanceFeatureExtraction: {'build': {'cell_seg_key': 'segmentation',
                                                  'channel_keys': ['serum_IgG', 'marker']}},
            ExtractBackground: {'build': {'serum_key': 'serum_IgG',
                                          'cell_seg_key': 'segmentation',
                                          'actual_channels_to_use': ['serum_IgG', 'marker'],
                                          'marker_key': 'marker'}},
            FindInfectedCells: {'build': {'cell_seg_key': 'segmentation'}},
            CellLevelQC: {'build': {'cell_seg_key': 'segmentation',
                                    'serum_key': 'serum_IgG',
                                    'outlier_criteria': outlier_criteria}}
        }
        run_workflow('test', self.folder, job_dict, input_folder=self.input_folder)

    def test_size_thresholds(self):
        # choose high low / low high thresholds to get sum hits
        min_size = 400
        max_size = 750
        outlier_criteria = {'min_cell_size': min_size,
                            'max_cell_size': max_size,
                            'min_nucleus_size': None,
                            'max_nucleus_size': None}
        self.run_wf2(outlier_criteria)

        # check the results
        for path in glob(os.path.join(self.folder, '*.h5')):
            qc_table_name = 'segmentation/serum_IgG_outliers'
            self.check_size_thresholds(path, 'segmentation', qc_table_name,
                                       min_size, max_size)


class TestImageLevelQC(unittest.TestCase):
    folder = './out'

    def get_image_path(self, image_id):
        name_pattern = 'WellC01_PointC01_000%i_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.h5'
        return os.path.join(self.folder, name_pattern % image_id)

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
        self.exp_mixed_results = {}
        manual_outlier = []

        for ii, seg in enumerate(self.segs):
            path = self.get_image_path(ii)
            name = os.path.splitext(os.path.split(path)[1])[0]

            if ii < 2:
                res = (0, 'none')
                res_mixed = (0, 'none', '0')
                manual_outlier.append([name, 0])
            elif ii == 2:
                res = (0, 'none')
                res_mixed = (1, 'none', '1')
                manual_outlier.append([name, 1])
            elif ii == 3:
                res = (1, 'too few cells')
                res_mixed = (1, 'too few cells', '0')
                manual_outlier.append([name, 0])
            elif ii == 4:
                res = (1, 'too many cells')
                res_mixed = (1, 'too many cells', '0')
                manual_outlier.append([name, 0])

            self.exp_heuristic_results[name] = res
            self.exp_mixed_results[name] = res_mixed

            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, seg)

            # make dummy images for the serum and marker channel
            dummy = np.zeros(seg.shape, dtype='float32')
            with open_file(path, 'a') as f:
                write_image(f, 'serum', dummy)
            with open_file(path, 'a') as f:
                write_image(f, 'marker', dummy)

        tagged_path = os.path.join(self.folder, 'out_tagger_state.csv')
        manual_outlier = pd.DataFrame(manual_outlier, columns=['filename', 'label'])
        manual_outlier.to_csv(tagged_path, index=False)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def run_wf(self, outlier_criteria, outlier_predicate=lambda im: -1):
        from batchlib.analysis.cell_level_analysis import (InstanceFeatureExtraction,
                                                           FindInfectedCells)
        from batchlib.analysis.cell_analysis_qc import ImageLevelQC
        from batchlib.workflow import run_workflow

        job_dict = {
            InstanceFeatureExtraction: {'build': {'cell_seg_key': self.cell_seg_key}},
            FindInfectedCells: {'build': {'cell_seg_key': self.cell_seg_key}},
            ImageLevelQC: {'build': {'cell_seg_key': self.cell_seg_key,
                                     'outlier_criteria': outlier_criteria,
                                     'outlier_predicate': outlier_predicate}}
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

    def test_qc_with_manual_outliers(self):
        from batchlib.outliers.outlier import OutlierPredicate
        outlier_criteria = {'min_number_cells': 10,
                            'max_number_cells': 1000}
        outlier_predicate = OutlierPredicate(self.folder, 'out')
        self.run_wf(outlier_criteria, outlier_predicate)

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

        keyword_manual = 'manual: '
        keyword_heuristic = 'heuristic: '
        exp_results = self.exp_mixed_results
        for im_name, (res_outlier, res_type) in results.items():
            self.assertIn(im_name, exp_results)
            exp_outlier, exp_type_heuristic, exp_type_manual = exp_results[im_name]
            self.assertEqual(res_outlier, exp_outlier)

            idx_heuristic = res_type.find(keyword_heuristic) + len(keyword_heuristic)
            res_type_heuristic = res_type[idx_heuristic:]
            self.assertEqual(res_type_heuristic, exp_type_heuristic)

            idx_manual = res_type.find(keyword_manual) + len(keyword_manual)
            idx_heuristic_start = res_type.find(keyword_heuristic)
            res_type_manual = res_type[idx_manual:idx_heuristic_start].rstrip()
            res_type_manual = res_type_manual.rstrip(';')
            self.assertEqual(res_type_manual, exp_type_manual)


class TestWellLevelQC(unittest.TestCase):
    n_wells = 8
    ims_per_well = 4
    folder = './out'
    shape = (32, 32)
    cell_seg_key = 'seg'

    def get_image_path(self, well_id, image_id):
        name_pattern = 'WellC0%i_PointC0%i_000%i_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0216.h5'
        return os.path.join(self.folder, name_pattern % (well_id, well_id, image_id))

    def get_well_name(self, well_id):
        return 'C0%i' % well_id

    def write_dummy_channels(self, path):
        dummy = np.zeros(self.shape, dtype='float32')
        with open_file(path, 'a') as f:
            write_image(f, 'serum', dummy)
        with open_file(path, 'a') as f:
            write_image(f, 'marker', dummy)

    # write a normal well that passes QC
    def normal_well(self, well_id):
        for im_id in range(self.ims_per_well):
            path = self.get_image_path(well_id, im_id)

            # write the segmentation and dummy images
            seg = np.random.randint(0, 100, size=self.shape, dtype='uint64')
            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, seg)
            self.write_dummy_channels(path)
        return 0, 'none'

    # write a well that has too few cells
    def few_cell_well(self, well_id):
        segs = [np.random.randint(0, 15, size=self.shape, dtype='uint64'),
                np.ones(self.shape, dtype='uint64'),
                np.ones(self.shape, dtype='uint64'),
                np.random.randint(0, 15, size=self.shape, dtype='uint64')]
        for im_id in range(self.ims_per_well):
            path = self.get_image_path(well_id, im_id)
            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, segs[im_id])
            self.write_dummy_channels(path)
        return 1, 'too few cells'

    # write a well that has too many cells
    def many_cell_well(self, well_id):
        size = np.prod(list(self.shape))
        segs = [np.random.randint(0, 480, size=self.shape, dtype='uint64'),
                np.arange(size, dtype='uint64').reshape(self.shape),
                np.arange(size, dtype='uint64').reshape(self.shape),
                np.random.randint(0, 480, size=self.shape, dtype='uint64')]
        for im_id in range(self.ims_per_well):
            path = self.get_image_path(well_id, im_id)
            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, segs[im_id])
            self.write_dummy_channels(path)
        return 1, 'too many cells'

    # write a well that has too few control cells
    def few_control_well(self, well_id):
        # TODO implement this !
        return self.normal_well(well_id)

    # write a well that has a low control cell ratio
    def low_control_fraction_well(self, well_id):
        # TODO implement this !
        return self.normal_well(well_id)

    # write a well that has negative ratios
    def negative_ratio_well(self, well_id):
        # TODO implement this !
        return self.normal_well(well_id)

    def all_outlier_image_well(self, well_id):
        segs = [np.ones(self.shape, dtype='uint64')] * self.ims_per_well
        for im_id in range(self.ims_per_well):
            path = self.get_image_path(well_id, im_id)
            with open_file(path, 'a') as f:
                write_image(f, self.cell_seg_key, segs[im_id])
            self.write_dummy_channels(path)
        return 1, 'too few cells'

    def setUp(self):
        os.makedirs(self.folder, exist_ok=True)

        self.exp_results = {}
        self.exp_results_simple = {}
        for well_id in range(self.n_wells):

            # write all outlier images well as first well,
            # to check for this corner case
            if well_id == 0:
                res = self.all_outlier_image_well(well_id)
                res_simple = res
            # next write all the other outliers
            elif well_id == 1:
                res = self.few_cell_well(well_id)
                res_simple = res
            elif well_id == 2:
                res = self.many_cell_well(well_id)
                res_simple = res
            elif well_id == 3:
                res = self.few_control_well(well_id)
                res_simple = (0, 'none')
            elif well_id == 4:
                res = self.low_control_fraction_well(well_id)
                res_simple = (0, 'none')
            elif well_id == 5:
                res = self.negative_ratio_well(well_id)
                res_simple = (0, 'none')
            # the other wells are non-outlies
            else:
                res = self.normal_well(well_id)
                res_simple = res

            well_name = self.get_well_name(well_id)
            self.exp_results[well_name] = res
            self.exp_results_simple[well_name] = res_simple

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def run_wf(self, outlier_criteria, run_analysis=False):
        from batchlib.analysis.cell_level_analysis import (InstanceFeatureExtraction,
                                                           FindInfectedCells)
        from batchlib.analysis.cell_analysis_qc import ImageLevelQC, WellLevelQC
        from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
        from batchlib.workflow import run_workflow

        job_dict = {
            InstanceFeatureExtraction: {'build': {'cell_seg_key': self.cell_seg_key}},
            FindInfectedCells: {'build': {'cell_seg_key': self.cell_seg_key}},
            ImageLevelQC: {'build': {'cell_seg_key': self.cell_seg_key}},
            WellLevelQC: {'build': {'cell_seg_key': self.cell_seg_key,
                                    'outlier_criteria': outlier_criteria}}
        }
        if run_analysis:
            job_dict[CellLevelAnalysis] = {'build': {'cell_seg_key': self.cell_seg_key}}
        run_workflow('test', self.folder, job_dict)

    def _run_test(self, outlier_criteria, exp_results, run_analysis=False):
        self.run_wf(outlier_criteria, run_analysis=run_analysis)

        path = os.path.join(self.folder, 'out_table.hdf5')
        self.assertTrue(os.path.exists(path))
        qc_table_name = 'wells/outliers'
        with open_file(path, 'r') as f:
            col_names, table = read_table(f, qc_table_name)

        n_cols = 3
        exp_shape = (self.n_wells, n_cols)

        self.assertEqual(len(col_names), n_cols)
        self.assertEqual(table.shape, exp_shape)

        well_name_col = col_names.index('well_name')
        outlier_col = col_names.index('is_outlier')
        outlier_type_col = col_names.index('outlier_type')
        results = {row[well_name_col]: (row[outlier_col], row[outlier_type_col])
                   for row in table}

        exp_results = self.exp_results_simple
        for well_name, (res_outlier, res_type) in results.items():
            self.assertIn(well_name, exp_results)
            exp_outlier, exp_type = exp_results[well_name]
            self.assertEqual(res_outlier, exp_outlier)
            self.assertEqual(res_type, exp_type)

    def test_qc_simple(self):
        outlier_criteria = {'max_number_cells_per_image': 500,
                            'min_number_cells_per_image': 10,
                            'min_number_control_cells_per_image': None,
                            'min_fraction_of_control_cells': None,
                            'check_ratios': None}
        self._run_test(outlier_criteria, self.exp_results_simple)

    def test_qc_integration(self):
        outlier_criteria = {'max_number_cells_per_image': 500,
                            'min_number_cells_per_image': 10,
                            'min_number_control_cells_per_image': None,
                            'min_fraction_of_control_cells': None,
                            'check_ratios': None}
        self._run_test(outlier_criteria, self.exp_results_simple, run_analysis=True)

    # TODO we need to generate infected and control cells for this to work
    @unittest.skip
    def test_qc(self):
        outlier_criteria = {'max_number_cells_per_image': 500,
                            'min_number_cells_per_image': 10,
                            'min_number_control_cells_per_image': 5,
                            'min_fraction_of_control_cells': 0.05,
                            'check_ratios': True}
        self._run_test(outlier_criteria, self.exp_results)


if __name__ == '__main__':
    unittest.main()
