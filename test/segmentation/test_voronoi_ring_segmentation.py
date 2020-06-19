import os
import unittest
from glob import glob
from shutil import rmtree

import numpy as np
from batchlib.util import read_image, open_file


class TestVoronoiRingSegmentation(unittest.TestCase):
    # input_folder = os.path.join(os.path.split(__file__)[0], '../../data/test_data/test')
    input_folder = os.path.join(os.path.split(__file__)[0], '../../data/test_data/naming_schemes/scheme1')
    misc_folder = os.path.join(os.path.split(__file__)[0], '../../misc')
    n_cpus = 4

    folder = './out'
    seg_key = 'nucleus_seg'
    seg_out_key = 'nucleus_seg_dilated'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def prepare(self):
        from batchlib.preprocessing import Preprocess
        from batchlib.segmentation.stardist_prediction import StardistPrediction
        from batchlib.workflow import run_workflow

        model_root = os.path.join(self.misc_folder, 'models/stardist')
        model_name = '2D_dsb2018'

        job_dict = {
            Preprocess.from_folder: {'build': {'input_folder': self.input_folder},
                                     'run': {'n_jobs': self.n_cpus}},
            StardistPrediction: {'build': {'model_root': model_root,
                                           'model_name': model_name,
                                           'input_key': 'nuclei',
                                           'output_key': self.seg_key},
                                 'run': {'gpu_id': None,
                                         'n_jobs': self.n_cpus}}
        }
        run_workflow('test', self.folder, job_dict, input_folder=self.input_folder)

    def check_result(self, remove_nucleus):
        files = glob(os.path.join(self.folder, '*.h5'))
        for ff in files:
            with open_file(ff, 'r') as f:
                seg = read_image(f, self.seg_key)
                res = read_image(f, self.seg_out_key)

            self.assertEqual(seg.shape, res.shape)
            seg_ids = np.unique(seg)
            res_ids = np.unique(res)
            self.assertTrue(np.array_equal(seg_ids, res_ids))

            seg_mask = seg > 0
            if remove_nucleus:
                self.assertTrue(np.allclose(res[seg_mask], 0))
            else:
                self.assertTrue(np.allclose(res[seg_mask], seg[seg_mask]))

    def test_voronoi_fixed_width(self):
        from batchlib.segmentation.voronoi_ring_segmentation import VoronoiRingSegmentation
        self.prepare()
        job = VoronoiRingSegmentation(self.seg_key, self.seg_out_key, ring_width=5)
        job(self.folder)
        self.check_result(True)

    def test_voronoi_keep_nucleus(self):
        from batchlib.segmentation.voronoi_ring_segmentation import VoronoiRingSegmentation
        self.prepare()
        job = VoronoiRingSegmentation(self.seg_key, self.seg_out_key, ring_width=5, remove_nucleus=False)
        job(self.folder)
        self.check_result(False)

    def test_voronoi_dynamic_width(self):
        from batchlib.segmentation.voronoi_ring_segmentation import VoronoiRingSegmentation
        self.prepare()
        job = VoronoiRingSegmentation(self.seg_key, self.seg_out_key, radius_factor=1.5)
        job(self.folder)
        self.check_result(True)


if __name__ == '__main__':
    unittest.main()
