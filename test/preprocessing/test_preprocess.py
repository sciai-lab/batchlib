import os
import unittest
from glob import glob
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
        from batchlib.preprocessing import Preprocess, get_channel_settings
        fname = glob(os.path.join(self.in_folder, '*.tiff'))[0]
        names, settings, reorder = get_channel_settings(fname)
        job = Preprocess(names, settings, reorder=reorder)
        job(self.folder, self.in_folder, n_jobs=4)

    def test_preprocess_with_barrel_correction(self):
        from batchlib.preprocessing import Preprocess, get_channel_settings
        fname = glob(os.path.join(self.in_folder, '*.tiff'))[0]
        names, settings, reorder = get_channel_settings(fname)

        barrel_corrector_path = '../../misc/barrel_corrector.h5'

        job = Preprocess(names, settings, reorder=reorder,
                         barrel_corrector_path=barrel_corrector_path)
        job(self.folder, self.in_folder, n_jobs=4)


if __name__ == '__main__':
    unittest.main()
