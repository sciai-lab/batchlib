import os
import unittest
from abc import ABC
from glob import glob
from shutil import rmtree

from batchlib.workflows import run_cell_analysis, cell_analysis_parser
from batchlib.workflows.mean_and_sum_cell_analysis import mean_and_sum_cell_analysis

ROOT = os.path.join(os.path.split(__file__)[0], '../data/test_data')


class BaseTestMixin(ABC):
    """ Base class for workflow integration tests.

    Deriving tests also need to inherit from unittest.TestCase
    and implement the following two functions:
    - get_parser(config_folder, config_name)
    - run_workflow(config)
    """
    folder = './out'
    use_unsafe_remove = False

    def _unsafe_rm(self):

        def continue_removing(fu, path, excinfo):
            pass

        # throws some error due to busy devices sometimes (I assume this is due to the locking)
        # so we just skip errors, all important files will be removed
        rmtree(self.folder, onerror=continue_removing)

    def tearDown(self):
        if self.use_unsafe_remove:
            # this is necessary on some nfs, but we shouldn't use it as default
            self._unsafe_rm()
        else:
            rmtree(self.folder)

    def _run_workflow(self, folder=None, input_folder=None, config_name='test_cell_analysis.conf'):
        parser = self.get_parser('./configs', config_name)
        config, _ = parser.parse_known_args()

        if folder is not None:
            config.folder = folder

        if input_folder is not None:
            config.input_folder = input_folder

        self.run_workflow(config)

    def test_workflow(self):
        self._run_workflow()
        # TODO check the results

    def _test_naming_scheme(self, in_folder):
        name = os.path.split(in_folder)[1]
        folder = os.path.join(self.folder, name)
        self._run_workflow(folder=folder, input_folder=in_folder)

    def test_workflow_for_naming_schemes(self):
        root = os.path.join(ROOT, 'naming_schemes')
        folders = glob(os.path.join(root, "*"))
        folders.sort()

        # for debugging
        # folders = folders[-1:]

        for folder in folders:
            self._test_naming_scheme(folder)
            # TODO check the results

    def test_fixed_background(self):
        folder = os.path.join(ROOT, 'naming_schemes', 'scheme4')
        self._run_workflow(input_folder=folder, config_name='test_cell_analysis_bg.conf')


class TestCellAnalysis(BaseTestMixin, unittest.TestCase):
    @staticmethod
    def get_parser(config_folder, config_name):
        return cell_analysis_parser(config_folder, config_name)

    @staticmethod
    def run_workflow(config):
        return run_cell_analysis(config)


class TestMeanAndSum(BaseTestMixin, unittest.TestCase):
    @staticmethod
    def get_parser(config_folder, config_name):
        return cell_analysis_parser(config_folder, config_name)

    @staticmethod
    def run_workflow(config):
        return mean_and_sum_cell_analysis(config)


if __name__ == '__main__':
    unittest.main()
