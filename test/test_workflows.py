import os
import unittest
from abc import ABC
from glob import glob
from shutil import rmtree

import numpy as np

from batchlib.workflows import run_cell_analysis, cell_analysis_parser
from batchlib.workflows.mean_and_sum_cell_analysis import mean_and_sum_cell_analysis
from batchlib.util import read_table, open_file, has_table

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
        return config.folder

    def test_workflow(self):
        res_folder = self._run_workflow()
        self.check_results(res_folder)

    def _test_naming_scheme(self, in_folder):
        name = os.path.split(in_folder)[1]
        folder = os.path.join(self.folder, name)
        res_folder = self._run_workflow(folder=folder, input_folder=in_folder)
        self.check_results(res_folder)

    def test_workflow_for_naming_schemes(self):
        root = os.path.join(ROOT, 'naming_schemes')
        folders = glob(os.path.join(root, "*"))
        folders.sort()

        # for debugging
        # folders = folders[-1:]

        for folder in folders:
            self._test_naming_scheme(folder)

    def test_fixed_background(self):
        folder = os.path.join(ROOT, 'naming_schemes', 'scheme4')
        res_folder = self._run_workflow(input_folder=folder, config_name='test_cell_analysis_bg.conf')
        self.check_results(res_folder)


class TestCellAnalysis(BaseTestMixin, unittest.TestCase):
    @staticmethod
    def get_parser(config_folder, config_name):
        return cell_analysis_parser(config_folder, config_name)

    @staticmethod
    def run_workflow(config):
        return run_cell_analysis(config)

    # TODO implement
    @staticmethod
    def check_results(folder):
        pass


class TestMeanAndSum(BaseTestMixin, unittest.TestCase):
    @staticmethod
    def get_parser(config_folder, config_name):
        return cell_analysis_parser(config_folder, config_name)

    @staticmethod
    def run_workflow(config):
        return mean_and_sum_cell_analysis(config)

    def check_results(self, folder):
        from batchlib.analysis.merge_tables import modify_column_names

        plate_name = os.path.split(folder)[1]
        table_file = os.path.join(folder, plate_name + '_table.hdf5')
        self.assertTrue(os.path.exists(table_file))

        feature_identifiers = ['mean', 'sum']

        for table_type in ('images', 'wells'):

            table_key_mean = f'{table_type}/default_mean'
            table_key_sum = f'{table_type}/default_sum'
            table_key_merged = f'{table_type}/default'
            with open_file(table_file, 'r') as f:
                self.assertTrue(has_table(f, table_key_mean))
                self.assertTrue(has_table(f, table_key_sum))
                self.assertTrue(has_table(f, table_key_merged))

                cols_mean, tab_mean = read_table(f, table_key_mean)
                cols_sum, tab_sum = read_table(f, table_key_sum)
                cols_merged, tab_merged = read_table(f, table_key_merged)

                self.assertEqual(len(cols_mean), len(cols_sum))
                self.assertEqual(len(cols_mean), len(cols_merged))
                cols_mean = modify_column_names(cols_mean, feature_identifiers)
                cols_sum = modify_column_names(cols_sum, feature_identifiers)
                self.assertEqual(cols_mean, cols_sum)
                self.assertEqual(cols_mean, cols_merged)

                self.assertEqual(tab_mean.shape, tab_sum.shape)
                self.assertEqual(tab_mean.shape, tab_merged.shape)

                mean_col_ids = [ii for ii, name in enumerate(cols_merged) if 'mean' in name]
                sum_col_ids = [ii for ii, name in enumerate(cols_merged) if 'sum' in name]
                self.assertEqual(len(set(mean_col_ids) - set(sum_col_ids)), len(mean_col_ids))

                remaining_col_ids = list(set(range(len(cols_merged))) - set(mean_col_ids + sum_col_ids))

                self.assertTrue(np.array_equal(tab_merged[:, mean_col_ids], tab_mean[:, mean_col_ids]))
                self.assertTrue(np.array_equal(tab_merged[:, sum_col_ids], tab_sum[:, sum_col_ids]))
                self.assertTrue(np.array_equal(tab_merged[:, remaining_col_ids], tab_mean[:, remaining_col_ids]))


if __name__ == '__main__':
    unittest.main()
