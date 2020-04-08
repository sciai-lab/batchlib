import unittest
from shutil import rmtree


class TestStardist(unittest.TestCase):
    in_folder = '../../data/test_preprocessed'
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_stardist_prediction(self):
        from batchlib.segmentation.stardist import StardistPrediction

        model_root = '/home/pape/Work/covid/antibodies-nuclei/stardist/models/pretrained'
        model_name = '2D_dsb2018'

        in_key = 'raw'
        out_key = 'pred'

        job = StardistPrediction(model_root, model_name, in_key, out_key,
                                 input_channel=0)
        job(self.folder, self.in_folder)


if __name__ == '__main__':
    unittest.main()
