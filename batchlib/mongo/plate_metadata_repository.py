import argparse
import urllib.parse

from pymongo import MongoClient

from batchlib.mongo.utils import ASSAY_METADATA
from batchlib.util import get_logger

logger = get_logger('Workflow.PlateMetadataRepository')

TEST_NAMES = ['ELISA IgG', 'ELISA IgA', 'ELISA IgM',
              'mpBio IgG', 'mpBio IgM',
              'Luminex', 'NT', 'Roche', 'Abbot',
              'Rapid test IgM', 'Rapid test IgG',
              'IF IgG', 'IF IgA', 'days_after_onset']


class PlateMetadataRepository:
    """
    Simple Monogo API used to get the positive (cohort: C) and control (cohort: B) wells for a given plate
    as well as Elisa test results if available.

    """

    def __init__(self, db):
        self.assay_metadata_collection = db[ASSAY_METADATA]

    def _get_wells(self, plate_name):
        plate_doc = self.assay_metadata_collection.find_one({"name": plate_name})
        if plate_doc is None:
            logger.info(f"No plate for name {plate_name} was found in the DB")
            return None

        return plate_doc["wells"]

    def _filter_wells(self, plate_name, predicate):
        wells = self._get_wells(plate_name)
        if wells is None:
            return None

        return set(
            map(
                # get only the well name
                lambda w: w['name'],
                # filter wells by predicate
                filter(predicate, wells)
            )
        )

    def get_control_wells(self, plate_name):
        """
        Get wells corresponding to the healthy controls (patient_type: B) for a given plate.

        Args:
            plate_name (str): name of the plate
        Returns:
            set of control wells
        """

        _filter = lambda w: w.get('patient_type', None) in ['B', 'A', 'X', 'Z', 'E', 'CMV', 'EBV']

        return self._filter_wells(plate_name, _filter)

    def get_positive_wells(self, plate_name):
        """
        Get wells corresponding to the positive cases (patient_type: C) for a given plate.

        Args:
            plate_name (str): name of the plate
        Returns:
            set of positive wells
        """
        _filter = lambda w: w.get('patient_type', None) == 'C'

        return self._filter_wells(plate_name, _filter)

    def get_cohort_ids(self, plate_name):
        """
        Returns a dict:
        {
            well1: cohort_id1,
            well2: cohort_id2,
            ...
        }
        """
        wells = self._get_wells(plate_name)
        if wells is None:
            return {}
        result = {}
        for well in wells:
            if 'cohort_id' in well:
                result[well['name']] = well['cohort_id']

        return result

    def get_elisa_results(self, plate_name, field_names=TEST_NAMES):
        """
        Returns a dict
        {
            well1: [test_value1, test_value2, ...],
            well2: [test_value1, test_value2, ...],
            ....
        }
        where `well*` is a well name and `test_value*` are values of a given test (see TEST_NAMES).
        """
        wells = self._get_wells(plate_name)
        if wells is None:
            return {}
        result = {}
        for well in wells:
            test_results = []
            for fn in field_names:
                v = None
                if fn in well:
                    v = well[fn]
                test_results.append(v)

            result[well['name']] = test_results

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PlateMetadataRepository')

    parser.add_argument('--host', type=str, help='IP of the MongoDB primary DB', required=True)
    parser.add_argument('--port', type=int, help='MongoDB port', default=27017)

    parser.add_argument('--user', type=str, help='MongoDB user', required=True)
    parser.add_argument('--password', type=str, help='MongoDB password', required=True)

    parser.add_argument('--db', type=str, help='Default database', default='covid')
    args = parser.parse_args()

    # escape username and password to be URL friendly
    username = urllib.parse.quote_plus(args.user)
    password = urllib.parse.quote_plus(args.password)

    mongodb_uri = f'mongodb://{username}:{password}@{args.host}:{args.port}/?authSource={args.db}'

    logger.info(
        f'Connecting to MongoDB instance: {args.host}:{args.port}, user: {args.user}, authSource: {args.db}')

    client = MongoClient(mongodb_uri)

    logger.info(f'Getting database: {args.db}')

    db = client[args.db]

    plate_metadata = PlateMetadataRepository(db)

    # sample query
    plate_name = '20200417_152052_943'
    control_wells = plate_metadata.get_control_wells(plate_name)
    print(control_wells)

    positive_wells = plate_metadata.get_positive_wells(plate_name)
    print(positive_wells)
