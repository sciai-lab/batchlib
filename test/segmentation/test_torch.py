import os
import unittest
from shutil import rmtree


class TestStardist(unittest.TestCase):
    in_folder = '../../data/test_preprocessed'
    folder = './out'

    def _tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_stardist_prediction(self):
        from batchlib.segmentation.torch_prediction import TorchPrediction
        from batchlib.segmentation.unet import UNet2D

        in_key = 'raw'
        out_key = ['foreground', 'boundaries']

        model_path = os.path.join('/g/kreshuk/pape/Work/covid/antibodies-nuclei',
                                  'unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch')
        model_class = UNet2D
        model_kwargs = {
            'in_channels': 1,
            'out_channels': 2,
            'f_maps': [32, 64, 128, 256, 512],
            'testing': True
        }

        job = TorchPrediction(in_key, out_key, model_path, model_class, model_kwargs,
                              input_channel=2)
        job(self.folder, self.in_folder, gpu_id=0, batch_size=4)


if __name__ == '__main__':
    unittest.main()
