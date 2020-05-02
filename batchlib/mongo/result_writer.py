import glob
import os
import urllib.parse
from datetime import datetime

import git
import h5py
from pymongo import MongoClient

from batchlib.base import BatchJobOnContainer
from batchlib.util import get_logger
from batchlib.util.io import read_table

ASSAY_ANALYSIS_RESULTS = 'immuno-assay-analysis-results'

logger = get_logger('Workflow.BatchJob.DbResultWriter')


def _get_table_names(f):
    table_names = set()

    def _visitor(name):
        if name.endswith('cells'):
            table_names.add(name[:-6])

    f['tables'].visit(_visitor)
    return list(table_names)


def _table_object(column_names, table):
    # document attributes cannot contain '.'
    column_names = [cn.replace('.', '_') for cn in column_names]
    return [
        dict(zip(column_names, table_row)) for table_row in table
    ]


def _get_result_tables(in_file):
    with h5py.File(in_file, 'r') as f:
        result_tables = {}
        for table_name in _get_table_names(f):
            column_names, table = read_table(f, table_name)
            result_tables[table_name] = _table_object(column_names, table)

        return result_tables


def _get_log_path(work_dir):
    logs = list(glob.glob(os.path.join(work_dir, '*.log')))
    assert len(logs) == 1
    return logs[0]


def _parse_workflow_duration(work_dir):
    """
    Reads workflow duration by parsing the first and the last log event in the log file and taking the time difference
    """
    log_path = _get_log_path(work_dir)
    with open(log_path, 'r') as fh:
        lines = list(fh)
        for first_log in lines:
            if 'INFO' in first_log:
                break

        for last_log in reversed(lines):
            if 'INFO' in last_log:
                break

        start = datetime.strptime(first_log.split('[')[0].strip(), '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(last_log.split('[')[0].strip(), '%Y-%m-%d %H:%M:%S')

        delta = end - start
        return delta.seconds


def _parse_workflow_name(work_dir):
    """
    Parses workflow name from the log file
    """
    log_path = _get_log_path(work_dir)
    logfile = os.path.split(log_path)[1]
    workflow_name = os.path.splitext(logfile)[0]
    return workflow_name


def _get_git_sha():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        logger.warning(f'Cannot get batchlib_version: {e}')
        return None


class DbResultWriter(BatchJobOnContainer):
    def __init__(self, username, password, host, port=27017, db_name='covid',
                 input_pattern='*.hdf5', **super_kwargs):
        super().__init__(input_pattern=input_pattern, **super_kwargs)

        username = urllib.parse.quote_plus(username)
        password = urllib.parse.quote_plus(password)

        mongodb_uri = f'mongodb://{username}:{password}@{host}:{port}/?authSource={db_name}'

        logger.info(f'Connecting to MongoDB instance: {host}:{port}, user: {username}, authSource: {db_name}')

        try:
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            logger.info(f'Server info: {client.server_info()}')
            self.db = client[db_name]
        except Exception as e:
            logger.warning(f'Connection failure: {e}. Skipping DbResultWriter job.')
            self.db = None

    def run(self, input_files, output_files, **kwargs):
        if self.db is None:
            return

        assert len(input_files) == 1, f'Expected a single table file, but {len(input_files)} was given'

        plate_name = os.path.split(self.folder)[1]

        input_file = input_files[0]

        result_tables = _get_result_tables(input_file)

        # this is a bit hacky: parsing the workflow name and execution duration from the log file in the work_dir,
        # but I don't see a better way to do it atm
        work_dir = os.path.join(self.folder, 'batchlib')
        result_object = {
            "created_at": datetime.now(),
            "workflow_name": _parse_workflow_name(work_dir),
            "workflow_duration": _parse_workflow_duration(work_dir),
            "plate_name": plate_name,
            "batchlib_version": _get_git_sha(),
            "result_tables": result_tables
        }

        # TODO: this should be an upsert: insert if not exist, update if (workflow_name, plate_name, batchlib_version) exist in the collection already
        self.db[ASSAY_ANALYSIS_RESULTS].insert_one(result_object)
