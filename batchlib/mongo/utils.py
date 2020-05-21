import glob
import json
import os
from datetime import datetime

from batchlib.outliers.outlier import OutlierPredicate, DEFAULT_OUTLIER_DIR
from batchlib.util import get_logger
from batchlib.util.cohort_parser import CohortIdParser
from batchlib.util.elisa_results_parser import ElisaResultsParser
from batchlib.util.io import image_name_to_site_name

ASSAY_METADATA = 'immuno-assay-metadata'
ASSAY_ANALYSIS_RESULTS = 'immuno-assay-analysis-results'

SUPPORTED_FORMATS = ['*.h5', '*.tif', '*.tiff']

logger = get_logger('Workflow.BatchJob.DbResultWriter.Utils')

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
        # take only image name without file extension
        im_file = os.path.splitext(im_file)[0]
        images.append(
            {
                "name": im_file,
                "well_name": well_name,
                "site_name": image_name_to_site_name(im_file),
                "outlier": outlier_predicate(im_file),
                "outlier_type": "manual"
            }
        )
    return images


def _create_wells(plate_name, plate_dir, cohort_id_parser, elisa_results_parser):
    outlier_predicate = OutlierPredicate(DEFAULT_OUTLIER_DIR, plate_name)
    plate_cohorts = cohort_id_parser.get_cohorts_for_plate(plate_name)

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
        cohort_id = plate_cohorts.get(well_name, None)
        patient_type = None
        IgG_value, IgA_value = None, None
        if cohort_id is not None:
            patient_type = cohort_id[0]
            IgG_value, IgA_value = elisa_results_parser.get_elisa_values(cohort_id)

        wells.append(
            {
                "name": well_name,
                # assume all wells are valid for now
                "outlier": 0,
                "outlier_type": "manual",
                "images": _create_images(well_name, well_files, outlier_predicate),
                "cohort_id": cohort_id,
                "patient_type": patient_type,
                "elisa_IgG": IgG_value,
                "elisa_IgA": IgA_value
            }
        )

    return wells


def create_plate_doc(plate_name, plate_dir):
    cohort_id_parser = CohortIdParser()
    elisa_results_parser = ElisaResultsParser()

    logger.info(f'Creating plate object for: {plate_name}')
    result = {
        "name": plate_name,
        "created_at": _parse_creation_time(plate_name),
        # assume all plates are valid for now
        "outlier": 0,
        "outlier_type": "manual",
        "channel_mapping": _load_channel_mapping(plate_dir),
        "wells": _create_wells(plate_name, plate_dir, cohort_id_parser, elisa_results_parser)
    }

    return result


def _get_table_names(f):
    table_names = set()

    def _visitor(name):
        if name.endswith('cells'):
            table_names.add(name[:-6])

    f['tables'].visit(_visitor)
    return list(table_names)