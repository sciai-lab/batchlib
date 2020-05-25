import tempfile
import unittest
import os

from batchlib.util.logger import get_logger, add_file_handler


class TestLogging(unittest.TestCase):
    def test_propagation(self):
        with tempfile.TemporaryDirectory() as work_dir:
            parent = get_logger('Workflow')
            add_file_handler(parent, work_dir, 'workflow1')

            # create child logger
            child = get_logger('Workflow.Job')
            child.warning('test log')

            # assert that log event was passed to the file handler
            log_file = os.path.join(work_dir, 'workflow1.log')
            with open(log_file) as f:
                num_lines = sum(1 for _ in f)

            assert num_lines == 1

    def test_log_rollover(self):
        with tempfile.TemporaryDirectory() as work_dir:
            logger = get_logger('Workflow')
            workflow_name = 'workflow1'

            add_file_handler(logger, work_dir, workflow_name)
            logger.warning('test1')
            # create new file handler
            add_file_handler(logger, work_dir, workflow_name)
            logger.warning('test2')
            # create yet another file handler
            add_file_handler(logger, work_dir, workflow_name)
            logger.warning('test3')

            log_path = os.path.join(work_dir, workflow_name + '.log')
            expected_log_files = [log_path, log_path + '.1', log_path + '.2']

            assert all(os.path.exists(fp) for fp in expected_log_files)
