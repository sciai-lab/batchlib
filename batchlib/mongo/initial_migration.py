import argparse
import urllib.parse

import pymongo
from pymongo import MongoClient

from batchlib.mongo.utils import ASSAY_METADATA, ASSAY_ANALYSIS_RESULTS
from batchlib.util import get_logger

logger = get_logger('MongoDB Migrator')


def create_indexes(db):
    logger.info(f'Creating indexes on {ASSAY_METADATA} and {ASSAY_ANALYSIS_RESULTS}')
    assay_metadata = db[ASSAY_METADATA]
    assay_results = db[ASSAY_ANALYSIS_RESULTS]
    # create necessary indexes
    assay_metadata.create_index([('name', pymongo.ASCENDING)], unique=True)
    # create unique compound index on (workflow_name, plate_name, batchlib_version), i.e. reject result objects
    # for which those 3 values already exist in the collection
    assay_results.create_index([
        ('workflow_name', pymongo.ASCENDING),
        ('plate_name', pymongo.ASCENDING),
        ('batchlib_version', pymongo.ASCENDING),
    ], unique=True)


def update_well_assessment(plate_name, well_assessments):
    # TODO: implement when we have this info in parseable format
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MongoDB migrator')

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

    logger.info(f'Connecting to MongoDB instance: {args.host}:{args.port}, user: {args.user}, authSource: {args.db}')

    client = MongoClient(mongodb_uri)

    logger.info(f'Getting database: {args.db}')

    create_indexes(client[args.db])
