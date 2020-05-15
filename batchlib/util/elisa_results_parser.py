import glob
import math
import os
import re

import pandas as pd

DEFAULT_EXCEL_DIR = os.path.join(os.path.split(__file__)[0], '../../misc/elisa')


def _validate_values(elisa_IgG, elisa_IgA):
    def _parse_value(v):
        if isinstance(v, str):
            if '>' in v:
                v = v[v.find('>') + 1:]
            try:
                return float(v)
            except Exception:
                return float('nan')

        if not (isinstance(v, float) or isinstance(v, int)):
            return float('nan')
        return float(v)

    elisa_IgG = _parse_value(elisa_IgG)
    elisa_IgA = _parse_value(elisa_IgA)

    # return None if both numbers are NaN
    if math.isnan(elisa_IgG) and math.isnan(elisa_IgA):
        return None

    return elisa_IgG, elisa_IgA


def _load_elisa_results(excel_file):
    # cohort_id pattern
    p = re.compile('[A-Z]\\d+')

    # load all sheets
    sheets = pd.read_excel(excel_file, sheet_name=None)
    results = {}
    # iterate over sheets
    for df in sheets.values():
        # iterate over rows
        for row in df.values:
            cohort_id, elisa_IgG, elisa_IgA = row[:3]
            if not isinstance(cohort_id, str):
                continue
            # if there is a cohort_id match
            if p.match(cohort_id) is not None:
                # convert to lowercase
                cohort_id = cohort_id.lower()
                elisa_tuple = _validate_values(elisa_IgG, elisa_IgA)
                if elisa_tuple is not None and cohort_id not in results:
                    results[cohort_id] = elisa_tuple
    return results


class ElisaResultsParser:
    def __init__(self, excel_dir=DEFAULT_EXCEL_DIR):
        self.elisa_results = {}
        # parse outliers
        for excel_file in glob.glob(os.path.join(excel_dir, '*.xlsx')):
            self.elisa_results.update(_load_elisa_results(excel_file))
