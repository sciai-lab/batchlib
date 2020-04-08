import unittest


class TestWorkflow(unittest.TestCase):
    in_folder = '../data/test_inputs'
    folder = './out'

    def test_instance_segmentation(self):
        from batchlib.preprocessing import Preprocess
        from batchlib.segmentation import BoundaryAndMaskPrediction, SeededWatershed
        from batchlib.segmentation.stardist import StardistPrediction
        from batchlib.workflow import run_workflow

        ilastik_bin = '/home/pape/Work/covid/antibodies-nuclei/ilastik/run_ilastik.sh'
        ilastik_project = '/home/pape/Work/covid/antibodies-nuclei/ilastik/boundaries_and_foreground.ilp'

        model_root = '/home/pape/Work/covid/antibodies-nuclei/stardist/models/pretrained'
        model_name = '2D_dsb2018'

        in_key = 'raw'
        bd_key = 'pmap'
        mask_key = 'mask'
        nuc_key = 'nuclei'
        seg_key = 'cells'

        job_dict = {
            Preprocess: {'run': {'reorder': False, 'n_jobs': 4}},
            BoundaryAndMaskPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                                  'ilastik_project': ilastik_project,
                                                  'input_key': in_key,
                                                  'boundary_key': bd_key,
                                                  'mask_key': mask_key}},
            StardistPrediction: {'build': {'model_root': model_root,
                                           'model_name': model_name,
                                           'input_key': in_key,
                                           'output_key': nuc_key,
                                           'input_channel': 0}},
            SeededWatershed: {'build': {'pmap_key': bd_key,
                                        'seed_key': nuc_key,
                                        'output_key': seg_key,
                                        'mask_key': mask_key}}
        }

        run_workflow('InstanceSegmentation', self.folder, job_dict, input_folder=self.in_folder)


if __name__ == '__main__':
    unittest.main()
