import os
import glob

DEFAULT_EXCEL_DIR = os.path.join(os.path.split(__file__)[0], '../../misc/elisa')


def _load_elisa_results(excel_file):
    # TODO: implement
    pass


class ElisaResultsParser:
    def __init__(self, excel_dir=DEFAULT_EXCEL_DIR):
        self.elisa_results = {}
        # parse outliers
        for excel_file in glob.glob(os.path.join(excel_dir, '*.xlsx')):
            self.elisa_results.update(_load_elisa_results(excel_file))
