import os
import unittest
from shutil import rmtree


class TestStardist(unittest.TestCase):
    in_folder = '../../data/test_data/test'
    folder = './out'
    root = '/home/pape/Work/covid/antibodies-nuclei'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def prepare(self):
        from batchlib import run_workflow
        from batchlib.preprocessing import Preprocess

        job_dict = {
            Preprocess.from_folder: {'build': {'input_folder': self.in_folder}}
        }

        run_workflow('Prepare', self.folder, job_dict,
                     input_folder=self.in_folder)

    def _test_prediction(self, gpu_id, batch_size, threshold_channels={}):
        from batchlib.segmentation.torch_prediction import TorchPrediction
        from batchlib.segmentation.unet import UNet2D
        self.prepare()

        in_key = 'serum'
        out_key = ['foreground', 'boundaries']

        model_path = os.path.join(self.root,
                                  'unet_segmentation',
                                  'sample_models',
                                  'fg_boundaries_best_checkpoint.pytorch')
        model_class = UNet2D
        model_kwargs = {
            'in_channels': 1,
            'out_channels': 2,
            'f_maps': [32, 64, 128, 256, 512],
            'testing': True
        }

        job = TorchPrediction(in_key, out_key, model_path, model_class, model_kwargs)
        job(self.folder, self.in_folder, gpu_id=gpu_id, batch_size=batch_size,
            threshold_channels=threshold_channels)

    def _test_gpu(self):
        self._test_prediction(0, batch_size=4)

    def test_cpu(self):
        self._test_prediction(None, batch_size=1)

    def test_threshold(self):
        self._test_prediction(None, batch_size=1, threshold_channels={0: .5})


if __name__ == '__main__':
    unittest.main()
