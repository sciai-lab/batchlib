import os
import urllib.parse
from datetime import datetime

import h5py
from pymongo import MongoClient

from batchlib.base import BatchJobOnContainer
from batchlib.mongo.import_cohort_ids import import_cohort_ids_for_plate
from batchlib.mongo.utils import ASSAY_ANALYSIS_RESULTS, ASSAY_METADATA, create_plate_doc, parse_workflow_duration, \
    parse_workflow_name, parse_plate_dir
from batchlib.util import get_logger, get_commit_id
from batchlib.util.cohort_parser import CohortIdParser
from batchlib.util.io import read_table

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

    def check_output(self, path, **kwargs):
        if self.db is None:
            return True
        # check if result document already exist for a given (batchlib_version, workflow_name, plate_name)
        work_dir = os.path.join(self.folder, 'batchlib')
        _filter = {
            "workflow_name": parse_workflow_name(work_dir),
            "plate_name": os.path.split(self.folder)[1],
            "batchlib_version": get_commit_id()
        }
        result = self.db[ASSAY_ANALYSIS_RESULTS].find_one(_filter)
        # return False if no entry in the DB
        return result is not None

    def validate_output(self, path, **kwargs):
        # the output is stored in the DB and it's assumed to be valid
        return True

    def run(self, input_files, output_files, **kwargs):
        if self.db is None:
            return

        plate_name = os.path.split(self.folder)[1]

        assert len(input_files) == 1, f"Expected a single table hdf5 file, but {len(input_files)} were given"
        input_file = input_files[0]
        result_tables = _get_analysis_tables(input_file)

        # this is a bit hacky: parsing the workflow name and execution duration from the log file in the work_dir,
        # but I don't see a better way to do it atm
        work_dir = os.path.join(self.folder, 'batchlib')
        result_object = {
            "created_at": datetime.now(),
            "workflow_name": parse_workflow_name(work_dir),
            "workflow_duration": parse_workflow_duration(work_dir),
            "plate_name": plate_name,
            "batchlib_version": get_commit_id(),
            "result_tables": result_tables
        }

        # we've reached this point, so there is either no result document for a given
        # (batchlib_versin, workflow_name, plate_name)
        # or there is one and we want to replace it (i.e. force_recompute=True)
        _filter = {
            "workflow_name": result_object["workflow_name"],
            "plate_name": result_object["plate_name"],
            "batchlib_version": result_object["batchlib_version"]
        }
        self.db[ASSAY_ANALYSIS_RESULTS].find_one_and_replace(_filter, result_object, upsert=True)

        # create plate metadata
        plate_dir = parse_plate_dir(work_dir, default_dir=self.folder)
        plate_doc = create_plate_doc(plate_name, plate_dir)
        self.db[ASSAY_METADATA].find_one_and_replace({"name": plate_name}, plate_doc, upsert=True)

        # update cohort ids if present for the plate
        cohort_id_parser = CohortIdParser()
        import_cohort_ids_for_plate(plate_name, self.db[ASSAY_METADATA], cohort_id_parser)
