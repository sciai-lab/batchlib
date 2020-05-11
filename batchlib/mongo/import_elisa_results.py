import argparse
import urllib.parse

from pymongo import MongoClient

from batchlib.mongo.utils import ASSAY_METADATA
from batchlib.util import get_logger
from batchlib.util.elisa_results_parser import ElisaResultsParser

logger = get_logger('Workflow.BatchJob.DbResultWriter')


def import_elisa_results(db):
    # create elisa results
    elisa_results_parser = ElisaResultsParser()
    # get metadata collection
    assay_metadata = db[ASSAY_METADATA]

    # iterate over all plates
    for plate_doc in assay_metadata.find({}):
        plate_name = plate_doc['plate_name']

        should_replace = False
        for well in plate_doc["wells"]:
            cohort_id = well.get("cohort_id", None)
            # make sure cohort_id matching is not case sensitive
            cohort_id = cohort_id.lower()

            if cohort_id in elisa_results_parser.elisa_results:
                logger.info(
                    f"Saving elisa results for plate: {plate_name}, well: {well['name']}, cohort_id: {cohort_id}")
                # mark doc to be replaced
                should_replace = True
                # add elisa results to the well
                IgG_value, IgA_value = elisa_results_parser.elisa_results[cohort_id]
                well['elisa_IgG'] = IgG_value
                well['elisa_IgA'] = IgA_value

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

    import_elisa_results(client[args.db])
