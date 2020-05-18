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
    # iterate over all plates for which cohort ids were provided
    for plate_name, cohort_ids in cohort_id_parser.well_cohort_ids.items():
        import_cohort_ids_for_plate(plate_name, assay_metadata, cohort_id_parser)


def import_cohort_ids_for_plate(plate_name, assay_metadata, cohort_id_parser):
    logger.info(f'Importing cohort ids for plate: {plate_name}')

    # fetch plate metadata from DB
    plate_doc = assay_metadata.find_one({"name": plate_name})
    if plate_doc is None:
        logger.warning(f"Plate {plate_name} not found in the DB")
        return

    wells = plate_doc["wells"]

    cohort_ids = cohort_id_parser.well_cohort_ids.get(plate_name, None)
    if cohort_ids is None:
        logger.info(f"No cohort ids info for plate: {plate_name}. Skipping cohort ids DB update")
        return

    for plate_row in cohort_ids:
        for well_name, cohort_id in plate_row:
            matching_well = list(filter(lambda w: w["name"] == well_name, wells))
            if len(matching_well) != 1:
                if cohort_id != 'unknown':
                    logger.warning(f"Well {well_name} with cohort id {cohort_id} not found in DB")
                continue

            matching_well = matching_well[0]
            # update cohort_id and patient_type
            matching_well["cohort_id"] = cohort_id
            matching_well["patient_type"] = cohort_id[0]

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
        {"patient_type": "P", "description": "recovered from Sars-CoV infection, roughly 150 samples, sampling ongoing"},
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
