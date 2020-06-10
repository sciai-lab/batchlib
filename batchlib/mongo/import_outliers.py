import argparse
import urllib.parse

from pymongo import MongoClient

from batchlib.mongo.utils import ASSAY_METADATA
from batchlib.outliers.outlier import OutlierPredicate
from batchlib.util import get_logger

logger = get_logger('OutlierImporter')


def import_outliers(db):
    # get metadata collection
    assay_metadata = db[ASSAY_METADATA]

    # iterate over all plates
    for plate_doc in assay_metadata.find({}):
        plate_name = plate_doc['name']
        # FIXME @wolny this needs to come from somewhere else now
        outlier_predicate = OutlierPredicate(DEFAULT_OUTLIER_DIR, plate_name)

        should_replace = False
        for well in plate_doc['wells']:
            for im in well['images']:
                im_file = im['name']
                outlier_current = outlier_predicate(im_file)
                outlier_previous = im['outlier']
                if outlier_current != outlier_previous:
                    # outlier status changed -> update and replace
                    logger.info(
                        f"Updating outlier status of {plate_name}/{im_file}: {outlier_previous} -> {outlier_current}")
                    im['outlier'] = outlier_current
                    should_replace = True

        if should_replace:
            assay_metadata.replace_one({"name": plate_name}, plate_doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MongoDB cohort_id importer')

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

    import_outliers(client[args.db])
