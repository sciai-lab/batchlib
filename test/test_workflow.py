import unittest


class TestPreprocess(unittest.TestCase):
    in_folder = '../data/test_inputs'
    folder = './out'

    def test_simple_workflow(self):
        from batchlib.preprocessing import Preprocess
        from batchlib.segmentation import IlastikPrediction
        from batchlib.workflow import run_workflow

        ilastik_bin = '/home/pape/Work/covid/antibodies-nuclei/ilastik/run_ilastik.sh'
        ilastik_project = '/home/pape/Work/covid/antibodies-nuclei/ilastik/local_infection.ilp'
        in_key = 'raw'
        out_key = 'pred'

        job_dict = {
            Preprocess: {'run': {'reorder': False, 'n_jobs': 4}},
            IlastikPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                          'ilastik_project': ilastik_project,
                                          'input_key': in_key,
                                          'output_key': out_key}}
        }

        run_workflow('Simple', self.folder, job_dict, input_folder=self.in_folder)


if __name__ == '__main__':
    unittest.main()
