import glob
import os
import re

import pandas as pd

SUPPORTED_WELL_ROWS = [c for c in 'ABCDEFGH']
DEFAULT_EXCEL_DIR = os.path.join(os.path.split(__file__)[0], '../../misc/cohort_ids')


def _contains_well_name(row):
    for cell in row:
        if cell in SUPPORTED_WELL_ROWS:
            return True
    return False


def _parse_well_name(row):
    for i, cell in enumerate(row):
        if cell in SUPPORTED_WELL_ROWS:
            return cell, i
    raise RuntimeError("Not a well-row")


def _parse_cohort_ids(row, well_ind):
    # cohort_id pattern
    p = re.compile('[A-Z]\\d+')

    result = []
    for i in range(well_ind + 1, len(row)):
        cohort_id = 'unknown'
        cell = row[i]
        if not isinstance(cell, str):
            cell = str(cell)
        if p.match(cell) is not None:
            cohort_id = cell
        result.append(cohort_id)
    return result


def _load_well_cohort_ids(excel_file):
    df = pd.read_excel(excel_file)
    results = []
    for row in df.values:
        if _contains_well_name(row):
            well_row_name, well_ind = _parse_well_name(row)
            cohort_ids = _parse_cohort_ids(row, well_ind)
            well_cohort_ids = []
            for i, cohort_id in enumerate(cohort_ids):
                well_num = i + 1
                well_num = str(well_num)
                if len(well_num) == 1:
                    well_num = '0' + well_num
                well_name = well_row_name + well_num
                well_cohort_ids.append((well_name, cohort_id))
                # we expect only 12 columns
                if i == 11:
                    break
            results.append(well_cohort_ids)
    return results


class CohortIdParser:
    def __init__(self, excel_dir=DEFAULT_EXCEL_DIR):
        self.well_cohort_ids = {}
        # parse outliers
        for excel_file in glob.glob(os.path.join(excel_dir, '*.xlsx')):
            assert '_final.xlsx' in excel_file
            plate_name = os.path.split(excel_file)[1]
            plate_name = plate_name[:plate_name.find('_final')]
            self.well_cohort_ids[plate_name] = _load_well_cohort_ids(excel_file)
