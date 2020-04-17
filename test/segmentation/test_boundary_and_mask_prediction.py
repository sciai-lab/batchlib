import os
import unittest
from shutil import rmtree


class TestBoundaryAndMaskPrediction(unittest.TestCase):
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

    def test_boundary_and_mask_prediction(self):
        from batchlib.segmentation import BoundaryAndMaskPrediction

        self.prepare()

        # TODO this should go in the repo!
        ilastik_bin = os.path.join(self.root, 'ilastik/run_ilastik.sh')
        ilastik_project = os.path.join(self.root, 'ilastik/boundaries_and_foreground.ilp')

        job = BoundaryAndMaskPrediction(ilastik_bin, ilastik_project,
                                        input_key=['nuclei', 'marker', 'serum'],
                                        boundary_key='boundaries', mask_key='mask')
        job(self.folder, self.in_folder)


if __name__ == '__main__':
    unittest.main()
