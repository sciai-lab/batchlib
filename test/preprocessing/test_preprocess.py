import os
import unittest
from glob import glob
from shutil import rmtree

THIS_FOLDER = os.path.split(__file__)[0]


# TODO check the processed data for correctness
class TestPreprocess(unittest.TestCase):
    in_root = os.path.join(THIS_FOLDER, '../../data/test_data')
    misc_folder = os.path.join(THIS_FOLDER, '../../misc')
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
        from batchlib.preprocessing import Preprocess, get_barrel_corrector
        in_folder = os.path.join(self.in_root, 'test')
        barrel_corrector_root = os.path.join(self.misc_folder, 'barrel_correctors')
        barrel_corrector_path = get_barrel_corrector(barrel_corrector_root, in_folder)
        job = Preprocess.from_folder(in_folder,
                                     barrel_corrector_path=barrel_corrector_path)
        job(self.folder, in_folder, n_jobs=4)

    def test_nameing_schemes(self):
        from batchlib.preprocessing import Preprocess
        root = os.path.join(self.in_root, 'naming_schemes')
        folders = glob(os.path.join(root, "*"))
        for in_folder in folders:
            job = Preprocess.from_folder(in_folder)
            job(self.folder, in_folder)

    def test_nameing_schemes_with_barrel_correction(self):
        from batchlib.preprocessing import Preprocess, get_barrel_corrector
        root = os.path.join(self.in_root, 'naming_schemes')
        barrel_corrector_root = os.path.join(self.misc_folder, 'barrel_correctors')
        folders = glob(os.path.join(root, "*"))
        for in_folder in folders:
            barrel_corrector_path = get_barrel_corrector(barrel_corrector_root, in_folder)
            job = Preprocess.from_folder(in_folder,
                                         barrel_corrector_path=barrel_corrector_path)
            job(self.folder, in_folder)


if __name__ == '__main__':
    unittest.main()
