import unittest
from shutil import rmtree


class TestBoundaryAndMaskPrediction(unittest.TestCase):
    in_folder = '../../data/test_preprocessed'
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_boundary_and_mask_prediction(self):
        from batchlib.segmentation import BoundaryAndMaskPrediction

        ilastik_bin = '/home/pape/Work/covid/antibodies-nuclei/ilastik/run_ilastik.sh'
        ilastik_project = '/home/pape/Work/covid/antibodies-nuclei/ilastik/boundaries_and_foreground.ilp'

        in_key = 'raw'
        bd_key = 'boundaries'
        mask_key = 'mask'

        job = BoundaryAndMaskPrediction(ilastik_bin, ilastik_project, in_key,
                                        bd_key, mask_key)
        job(self.folder, self.in_folder)


if __name__ == '__main__':
    unittest.main()
