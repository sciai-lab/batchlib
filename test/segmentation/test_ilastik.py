import unittest


class TestIlastik(unittest.TestCase):
    in_folder = '../../data/test_preprocessed'
    folder = './out'

    def test_ilastik(self):
        from batchlib.segmentation import IlastikPrediction

        ilastik_bin = '/home/pape/Work/covid/antibodies-nuclei/ilastik/run_ilastik.sh'
        ilastik_project = '/home/pape/Work/covid/antibodies-nuclei/ilastik/local_infection.ilp'

        in_key = 'raw'
        out_key = 'pred'

        n_jobs = 1
        job = IlastikPrediction(ilastik_bin, ilastik_project, in_key, out_key)
        job(self.folder, self.in_folder, n_jobs=n_jobs)


if __name__ == '__main__':
    unittest.main()
