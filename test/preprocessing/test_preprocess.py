import unittest


class TestPreprocess(unittest.TestCase):
    in_folder = '../../data/test_inputs'
    folder = './out'

    def test_preprocess(self):
        from batchlib.preprocessing import Preprocess
        job = Preprocess()
        job(self.folder, self.in_folder, reorder=False, n_jobs=4)


if __name__ == '__main__':
    unittest.main()
