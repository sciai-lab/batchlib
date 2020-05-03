import glob
import os
import urllib.parse
from datetime import datetime

import h5py
from pymongo import MongoClient

from batchlib.base import BatchJobOnContainer
from batchlib.util import get_logger, get_commit_id
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


def _get_analysis_name(in_file):
    filename = os.path.split(in_file)[1]
    # remove .hdf5 extension
    return os.path.splitext(filename)[0]


def _get_analysis_tables(in_file):
    with h5py.File(in_file, 'r') as f:
        tables = []
        for table_name in _get_table_names(f):
            column_names, table = read_table(f, table_name)
            tables.append(
                {
                    "table_name": table_name,
                    "results": _table_object(column_names, table)
                }
            )

        return tables


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


class DbResultWriter(BatchJobOnContainer):
    def __init__(self, username, password, host, port=27017, db_name='covid', **super_kwargs):
        super().__init__(input_pattern='*.hdf5', **super_kwargs)

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

    def check_output(self, path):
        if self.db is None:
            return True
        # check if result document already exist for a given (batchlib_version, workflow_name, plate_name)
        work_dir = os.path.join(self.folder, 'batchlib')
        _filter = {
            "workflow_name": _parse_workflow_name(work_dir),
            "plate_name": os.path.split(self.folder)[1],
            "batchlib_version": get_commit_id()
        }
        result = self.db[ASSAY_ANALYSIS_RESULTS].find_one(_filter)
        # return False if no entry in the DB
        return result is not None

    def validate_output(self, path):
        # the output is stored in the DB and it's assumed to be valid
        return True

    def run(self, input_files, output_files, **kwargs):
        if self.db is None:
            return

        plate_name = os.path.split(self.folder)[1]

        result_tables = []
        for input_file in input_files:
            analysis_name = _get_analysis_name(input_file)
            analysis_tables = _get_analysis_tables(input_file)
            result_tables.append(
                {
                    "analysis_name": analysis_name,
                    "tables": analysis_tables
                }
            )

        # this is a bit hacky: parsing the workflow name and execution duration from the log file in the work_dir,
        # but I don't see a better way to do it atm
        work_dir = os.path.join(self.folder, 'batchlib')
        result_object = {
            "created_at": datetime.now(),
            "workflow_name": _parse_workflow_name(work_dir),
            "workflow_duration": _parse_workflow_duration(work_dir),
            "plate_name": plate_name,
            "batchlib_version": get_commit_id(),
            "result_tables": result_tables
        }

        # we've reached this point, so there is either no result document for a given (batchlib_versin, workflow_name, plate_name)
        # or there is one and we want to replace it (i.e. force_recompute=True)
        _filter = {
            "workflow_name": result_object["workflow_name"],
            "plate_name": result_object["plate_name"],
            "batchlib_version": result_object["batchlib_version"]
        }
        self.db[ASSAY_ANALYSIS_RESULTS].find_one_and_replace(_filter, result_object, upsert=True)
