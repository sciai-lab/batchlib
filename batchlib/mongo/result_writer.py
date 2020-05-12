import os
import urllib.parse
from datetime import datetime
import time

import h5py
from pymongo import MongoClient

from batchlib.base import BatchJobOnContainer
from batchlib.mongo.import_cohort_ids import import_cohort_ids_for_plate
from batchlib.mongo.utils import ASSAY_ANALYSIS_RESULTS, ASSAY_METADATA, create_plate_doc, parse_workflow_duration, \
    parse_plate_dir
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


def _get_analysis_params(result_tables):
    analysis_params = list(filter(lambda t: t.get('table_name', None) == 'plate/analysis_parameter', result_tables))
    if not analysis_params:
        logger.warning(f'Cannot find plate/analysis_parameter table')
        return None
    else:
        analysis_params = analysis_params[0]
        return analysis_params["results"][0]


class DbResultWriter(BatchJobOnContainer):
    def __init__(self, workflow_name, input_folder, username, password, host, port=27017, db_name='covid',
                 **super_kwargs):
        super().__init__(input_pattern='*.hdf5', **super_kwargs)

        assert workflow_name is not None
        assert input_folder is not None

        self.workflow_name = workflow_name
        self.input_folder = input_folder

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
        _filter = {
            "workflow_name": self.workflow_name,
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

        result_object = {
            "created_at": datetime.now(),
            "workflow_name": self.workflow_name,
            "workflow_duration": self._get_workflow_duration(kwargs),
            "plate_name": plate_name,
            "batchlib_version": get_commit_id(),
            "analysis_parameters": _get_analysis_params(result_tables),
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

        # scan the plate directory, create the metadata document and insert into DB (or replace existing)
        plate_doc = create_plate_doc(plate_name, self.input_folder)
        self.db[ASSAY_METADATA].find_one_and_replace({"name": plate_name}, plate_doc, upsert=True)

        # update cohort ids if present for the plate
        cohort_id_parser = CohortIdParser()
        import_cohort_ids_for_plate(plate_name, self.db[ASSAY_METADATA], cohort_id_parser)

    @staticmethod
    def _get_workflow_duration(kwargs):
        t0 = kwargs.get('t0', None)
        if t0 is None:
            return 0
        return int(time.time() - t0)
