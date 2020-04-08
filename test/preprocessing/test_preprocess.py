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
        job = Preprocess()
        job(self.folder, self.in_folder, reorder=False, n_jobs=4)


if __name__ == '__main__':
    unittest.main()
