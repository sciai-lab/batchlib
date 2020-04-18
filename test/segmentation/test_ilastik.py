import os
import unittest
from shutil import rmtree


class TestIlastik(unittest.TestCase):
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

    def _run_test(self, n_jobs, n_threads=None):
        from batchlib.segmentation import IlastikPrediction
        self.prepare()

        # TODO this should go in the repo!
        ilastik_bin = os.path.join(self.root, 'ilastik/run_ilastik.sh')
        ilastik_project = os.path.join(self.root, 'ilastik/local_infection.ilp')

        out_key = 'pred'
        job = IlastikPrediction(ilastik_bin, ilastik_project,
                                input_key=['nuclei', 'marker', 'serum'],
                                output_key=out_key)
        job(self.folder, self.in_folder, n_jobs=n_jobs, n_threads=n_threads)

    def test_ilastik_single_job(self):
        self._run_test(1)

    def test_ilastik_multiple_job(self):
        self._run_test(4, n_threads=1)


if __name__ == '__main__':
    unittest.main()
