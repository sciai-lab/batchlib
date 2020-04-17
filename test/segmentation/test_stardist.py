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

    def test_stardist_prediction(self):
        from batchlib.segmentation.stardist_prediction import StardistPrediction
        self.prepare()

        model_root = os.path.join(self.root, 'stardist/models/pretrained')
        model_name = '2D_dsb2018'

        in_key = 'nuclei'
        out_key = 'pred'

        job = StardistPrediction(model_root, model_name, in_key, out_key)
        job(self.folder, self.in_folder)


if __name__ == '__main__':
    unittest.main()
