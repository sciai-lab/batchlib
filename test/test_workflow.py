import os
import unittest
from shutil import rmtree
from batchlib.segmentation.unet import UNet2D


class TestWorkflow(unittest.TestCase):
    in_folder = os.path.join(os.path.split(__file__)[0], '../data/test_data/test')
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_pixel_analysis_workflow(self):
        pass

    def test_cell_analysis_workflow(self):
        from batchlib.workflows import run_cell_analysis, cell_analysis_parser
        # TODO rename this config
        parser = cell_analysis_parser('./configs', 'test_cell_analysis.conf')
        config, _ = parser.parse_known_args()
        run_cell_analysis(config)
        # TODO check the results

    def test_small_workflow(self):
        from batchlib.preprocessing import Preprocess
        from batchlib.segmentation import SeededWatershed
        from batchlib.segmentation.stardist_prediction import StardistPrediction
        from batchlib.segmentation.torch_prediction import TorchPrediction
        from batchlib.workflow import run_workflow

        n_jobs = 4
        batch_size = 1

        this_folder = os.path.split(__file__)[0]
        model_root = os.path.join(this_folder, '../misc/models/stardist')
        model_name = '2D_dsb2018'

        torch_model_path = os.path.join(this_folder,
                                        '../misc/models/torch/fg_and_boundaries_V1.torch')
        torch_model_class = UNet2D
        torch_model_kwargs = {
            'in_channels': 1,
            'out_channels': 2,
            'f_maps': [32, 64, 128, 256, 512],
            'testing': True
        }

        serum_key = 'serum'
        nuclei_key = 'nuclei'

        bd_key = 'pmap'
        mask_key = 'mask'
        nuc_seg_key = 'nucleus_seg'
        cell_seg_key = 'cell_seg'

        job_dict = {
            Preprocess.from_folder: {'build': {'input_folder': self.in_folder},
                                     'run': {'n_jobs': n_jobs}},
            TorchPrediction: {'build': {'input_key': serum_key,
                                        'output_key': [mask_key, bd_key],
                                        'model_path': torch_model_path,
                                        'model_class': torch_model_class,
                                        'model_kwargs': torch_model_kwargs},
                              'run': {'gpu_id': None,
                                      'batch_size': batch_size,
                                      'threshold_channels': {0: 0.5}}},
            StardistPrediction: {'build': {'model_root': model_root,
                                           'model_name': model_name,
                                           'input_key': nuclei_key,
                                           'output_key': nuc_seg_key}},
            SeededWatershed: {'build': {'pmap_key': bd_key,
                                        'seed_key': nuc_seg_key,
                                        'output_key': cell_seg_key,
                                        'mask_key': mask_key}}
        }

        run_workflow('InstanceSegmentation', self.folder, job_dict,
                     input_folder=self.in_folder)


if __name__ == '__main__':
    unittest.main()
