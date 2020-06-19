import argparse
import urllib.parse

from pymongo import MongoClient

from batchlib.mongo.utils import ASSAY_METADATA
from batchlib.util import get_logger
from batchlib.util.cohort_parser import CohortIdParser

logger = get_logger('CohortImporter')


def import_cohort_ids(db):
    # parse excel files containing the cohort id for each well
    cohort_id_parser = CohortIdParser()

    # get metadata collection
    assay_metadata = db[ASSAY_METADATA]
    # iterate over all plates in the DB
    for plate_doc in assay_metadata.find({}):
        plate_name = plate_doc['name']
        logger.info(f'Importing cohort ids for plate: {plate_name}')

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate(plate_name)

        if not plate_cohorts:
            logger.warning(f"No cohort metadata for plate: {plate_name}. Check your cohort excel files.")
            continue

        for well in plate_doc["wells"]:
            cohort_id = plate_cohorts[well['name']]
            well["cohort_id"] = cohort_id
            well["patient_type"] = cohort_id[0]

        # replace plate with cohort info update
        assay_metadata.replace_one({"name": plate_name}, plate_doc)


def import_cohort_descriptions(db):
    cohort_descriptions = [
        {"patient_type": "A",
         "description": "from 2015-16, people who had a common cold Corona infection at least 3 months before, 65 samples"},
        {"patient_type": "B", "description": "from 2018, healthy controls ca. 110 samples"},
        {"patient_type": "C",
         "description": "from patients in the hospital, all Sars-CoV positive and symptomatic at different stages post symptom onset; currently roughly 150 analyzed, sampling ongoing"},
        {"patient_type": "K",
         "description": "childrens study Heidelberg, roughly 1100 samples, sampling ongoing (goal 3300)"},
        {"patient_type": "M",
         "description": "from 2020, not tested for RNA or negative, roughly 150 samples, sampling ongoing"},
        {"patient_type": "P",
         "description": "recovered from Sars-CoV infection, roughly 150 samples, sampling ongoing"},
        {"patient_type": "X", "description": "from before 2018, patients with mycoplasma infection"}
    ]

    db["cohort-descriptions"].delete_many({})
    db["cohort-descriptions"].insert_many(cohort_descriptions)


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

    import_cohort_ids(client[args.db])

    import_cohort_descriptions(client[args.db])
