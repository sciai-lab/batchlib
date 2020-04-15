import unittest
from shutil import rmtree


class TestPreprocess(unittest.TestCase):
    in_folder = '../../data/test_inputs'
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_preprocess(self):
        from batchlib.preprocessing import Preprocess
        job = Preprocess.from_folder(self.in_folder)
        job(self.folder, self.in_folder, n_jobs=4)

    def test_preprocess_with_barrel_correction(self):
        from batchlib.preprocessing import Preprocess
        barrel_corrector_path = '../../misc/barrel_corrector.h5'
        job = Preprocess.from_folder(self.in_folder, barrel_corrector_path=barrel_corrector_path)
        job(self.folder, self.in_folder, n_jobs=4)


if __name__ == '__main__':
    unittest.main()
