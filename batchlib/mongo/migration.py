import argparse
import json
from datetime import datetime
import glob

import pymongo
from pymongo import MongoClient

from batchlib.outliers.outlier import OutlierPredicate, DEFAULT_OUTLIER_DIR
from batchlib.util import get_logger
import urllib.parse
import os

ASSAY_METADATA = 'immuno-assay-metadata'
ASSAY_ANALYSIS_RESULTS = 'immuno-assay-analysis-results'

SUPPORTED_FORMATS = ['*.tif', '*.tiff']

logger = get_logger('MongoDB Migrator')

default_channel_mapping = {
    "DAPI": "nuclei",
    "WF_GFP": "marker",
    "TRITC": "serum"
}


def _parse_creation_time(plate_name):
    parts = plate_name.split('_')
    date_ind = -1

    for i, p in enumerate(parts):
        if p.startswith('202'):
            date_ind = i
            break

    if date_ind == -1:
        logger.info(f'Cannot parse date from plate {plate_name}')
        return datetime.now()

    _date = parts[i]
    _time = parts[i + 1]

    year = int(_date[:4])
    month = int(_date[4:6])
    day = int(_date[6:8])

    try:
        hour = int(_time[:2])
        minute = int(_time[2:4])
        second = int(_time[4:6])
    except ValueError:
        hour = minute = second = 0

    return datetime(year, month, day, hour, minute, second)


def _load_channel_mapping(plate_dir):
    file_path = os.path.join(plate_dir, 'channel_mapping.json')
    if not os.path.exists(file_path):
        logger.warning(f'No channel mapping: {file_path}. Using default.')
        return default_channel_mapping

    with open(file_path) as json_file:
        return json.load(json_file)


def _create_images(well_name, well_files, outlier_predicate):
    images = []
    for im_file in well_files:
        images.append(
            {
                "name": im_file,
                "well_name": well_name,
                "outlier": outlier_predicate(im_file),
                "outlier_type": "manual"
            }
        )
    return images


def _create_wells(plate_name, plate_dir):
    outlier_predicate = OutlierPredicate(DEFAULT_OUTLIER_DIR, plate_name)

    file_names = []
    for ext in SUPPORTED_FORMATS:
        file_names.extend(glob.glob(os.path.join(plate_dir, ext)))

    file_names = [os.path.split(fn)[1] for fn in file_names]
    well_dict = {}
    for filename in file_names:
        well_name = filename.split('_')[0][4:]
        well_files = well_dict.get(well_name, [])
        well_files.append(filename)
        well_dict[well_name] = well_files

    wells = []
    for well_name, well_files in well_dict.items():
        wells.append(
            {
                "name": well_name,
                # assume all wells are valid for now
                "outlier": 0,
                "outlier_type": "manual",
                "manual_assessment": "unknown",
                "images": _create_images(well_name, well_files, outlier_predicate)
            }
        )

    return wells


def _create_plate_doc(plate_name, plate_dir):
    logger.info(f'Creating plate object for: {plate_name}')
    result = {
        "name": plate_name,
        "created_at": _parse_creation_time(plate_name),
        # assume all plates are valid for now
        "outlier": 0,
        "outlier_type": "manual",
        "channel_mapping": _load_channel_mapping(plate_dir),
        "wells": _create_wells(plate_name, plate_dir)
    }

    return result


def migrate(input_folder, db):
    logger.info(f'Migrating plates from: {input_folder}...')
    # get assay metadata collection
    assay_metadata = db[ASSAY_METADATA]
    # create necessary indexes
    assay_metadata.create_index([('name', pymongo.ASCENDING)], unique=True)

    plate_docs = []
    for filename in os.listdir(input_folder):
        plate_name = filename
        plate_dir = os.path.join(input_folder, plate_name)
        if os.path.isdir(plate_dir):
            plate_doc = _create_plate_doc(plate_name, plate_dir)
            if plate_doc is not None:
                plate_docs.append(plate_doc)

    # import plates
    logger.info(f'Importing {len(plate_docs)} plates')
    assay_metadata.insert_many(plate_docs)


def update_well_assessment(plate_name, well_assessments):
    # TODO: implement when we have this info in parseable format
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MongoDB migrator')
    parser.add_argument('--input_folder', type=str, help='Path to the directory containing all the plates',
                        required=True)

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

    db = client[args.db]

    migrate(args.input_folder, db)
