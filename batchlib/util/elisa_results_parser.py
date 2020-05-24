import glob
import math
import os

import pandas as pd

from batchlib.util.cohort_parser import COHORT_PATTERNS

DEFAULT_EXCEL_DIR = os.path.join(os.path.split(__file__)[0], '../../misc/elisa')

SUPPORTED_FILED_NAMES = [
    'ELISA IgG', 'ELISA IgA', 'ELISA IgM',
    'mpBio IgG', 'mpBio IgM',
    'Luminex', 'NT', 'Roche', 'Abbot',
    'Rapid test IgM', 'Rapid test IgG',
    'IF IgG', 'IF IgA',
    'cohort_id'
]


def _validate_elisa_value(elisa_value):
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

    elisa_value = _parse_value(elisa_value)

    # return None if both numbers are NaN
    if math.isnan(elisa_value):
        return None

    return elisa_value


def _default_converter(value):
    return value


VALUE_CONVERTERS = {
    'ELISA IgG': _validate_elisa_value,
    'ELISA IgA': _validate_elisa_value,
    'ELISA IgM': _validate_elisa_value
}


def _parse_cohort_char(name):
    for c in name:
        if c.isupper():
            return c
    raise ValueError(f'Cannot parse cohort character from {name}')


def _find_field_indices(field_names):
    field_indices = {}
    for i, col in enumerate(field_names):
        if col in SUPPORTED_FILED_NAMES:
            field_indices[col] = i
        # sometimes cohort_id column is marked with the 'Nr' string
        if isinstance(col, str) and 'Nr' in col:
            field_indices['cohort_id'] = i

    if field_indices:
        # check if 'cohort_id' is present, if not choose the first column
        if not 'cohort_id' in field_indices:
            field_indices['cohort_id'] = 0

        return field_indices

    raise RuntimeError(f'Cannot parse any of the fields from sheet. Supported fields: {SUPPORTED_FILED_NAMES}')


def _parse_test_results(row, field_indices, cohort_char):
    cohort_id = row[field_indices['cohort_id']]
    if cohort_id is None:
        return None
    if isinstance(cohort_id, float) and math.isnan(cohort_id):
        return None

    # sometimes cohort_id is just a number and it has to be appended to cohort_char
    if isinstance(cohort_id, int):
        cohort_id = cohort_char + str(cohort_id)

    # check if we match the cohort pattern
    if any(p.match(cohort_id) is not None for p in COHORT_PATTERNS):
        result = {'cohort_id': cohort_id}
        for field_name, field_index in field_indices.items():
            if field_name in result:
                continue
            value_converter = VALUE_CONVERTERS.get(field_name, _default_converter)
            result[field_name] = value_converter(row[field_index])

        return result
    else:
        return None


def _load_elisa_results(excel_file):
    # load all sheets
    sheets = pd.read_excel(excel_file, sheet_name=None)
    results = {}
    # iterate over sheets
    for sheet_name, df in sheets.items():
        # parse cohort letter, e.g. A, B, C, P, K...
        cohort_char = _parse_cohort_char(sheet_name)
        field_indices = _find_field_indices(list(df.axes[1]))

        # iterate over rows
        for row in df.values:
            # parse test results from the row (Elisa, Roche, Abbot, Luminex, ...)
            test_results = _parse_test_results(row, field_indices, cohort_char)

            if test_results is not None:
                cohort_id = test_results['cohort_id']
                cohort_id = cohort_id.lower()
                if cohort_id not in results:
                    results[cohort_id] = test_results

    return results


class ElisaResultsParser:
    def __init__(self, excel_dir=DEFAULT_EXCEL_DIR):
        self.elisa_results = {}
        # parse outliers
        for excel_file in glob.glob(os.path.join(excel_dir, '*.xlsx')):
            self.elisa_results.update(_load_elisa_results(excel_file))

    def get_elisa_values(self, cohort_id, test_names=None):
        # test_names is None, return all possible values
        if test_names is None:
            test_names = list(SUPPORTED_FILED_NAMES)

        assert isinstance(cohort_id, str)
        cohort_id = cohort_id.lower()
        test_results = self.elisa_results.get(cohort_id)
        results = []
        for k in test_names:
            if k in test_results:
                results.append(test_results[k])
            else:
                results.append(None)

        return results
