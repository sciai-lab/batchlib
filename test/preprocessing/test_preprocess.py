import os
import unittest
from glob import glob
from shutil import rmtree


# TODO check for processed data for correctness
class TestPreprocess(unittest.TestCase):
    in_root = os.path.join(os.path.split(__file__)[0],
                           '../../data/test_data')
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def test_preprocess(self):
        from batchlib.preprocessing import Preprocess
        in_folder = os.path.join(self.in_root, 'test')
        job = Preprocess.from_folder(in_folder)
        job(self.folder, in_folder, n_jobs=4)

    def test_preprocess_with_barrel_correction(self):
        from batchlib.preprocessing import Preprocess
        in_folder = os.path.join(self.in_root, 'test')
        barrel_corrector_path = '../../misc/barrel_corrector.h5'
        job = Preprocess.from_folder(in_folder,
                                     barrel_corrector_path=barrel_corrector_path)
        job(self.folder, in_folder, n_jobs=4)

    def test_nameing_schemes(self):
        from batchlib.preprocessing import Preprocess
        root = os.path.join(self.in_root, 'naming_schemes')
        folders = glob(os.path.join(root, "*"))
        for folder in folders:
            name = os.path.split(folder)[1]
            out_folder = os.path.join(folder, name)
            job = Preprocess.from_folder(folder)
            job(out_folder, folder, n_jobs=4)


if __name__ == '__main__':
    unittest.main()
