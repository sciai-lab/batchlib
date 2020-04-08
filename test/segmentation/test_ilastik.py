import unittest
from shutil import rmtree


class TestIlastik(unittest.TestCase):
    in_folder = '../../data/test_preprocessed'
    folder = './out'

    def tearDown(self):
        try:
            rmtree(self.folder)
        except OSError:
            pass

    def _run_test(self, n_jobs):
        from batchlib.segmentation import IlastikPrediction

        ilastik_bin = '/home/pape/Work/covid/antibodies-nuclei/ilastik/run_ilastik.sh'
        ilastik_project = '/home/pape/Work/covid/antibodies-nuclei/ilastik/local_infection.ilp'

        in_key = 'raw'
        out_key = 'pred'

        job = IlastikPrediction(ilastik_bin, ilastik_project, in_key, out_key)
        job(self.folder, self.in_folder, n_jobs=n_jobs)

    def test_ilastik_single_job(self):
        self._run_test(1)

    @unittest.skip("Broken")
    def test_ilastik_multiple_job(self):
        self._run_test(4)


if __name__ == '__main__':
    unittest.main()
