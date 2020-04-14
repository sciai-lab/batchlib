import csv
import glob
import os


class Outliers:
    def __init__(self, root_table_dir):
        """
        Parses CSV files stored in 'root_table_dir', where each CSV corresponds to a given plate, and stores
        the results in a dictionary of the form {PLATE_NAME: TAGGER_STATE}, where TAGGER_STATE is a dict
        containing all images in a given PLATE_NAME together with their labels: (0 - accepted, 1 - outlier, -1 - skipped)
        """

        self.outliers = {}

        for csv_file in glob.glob(os.path.join(root_table_dir, '*.csv')):
            assert '_tagger_state.csv' in csv_file
            plate_name = os.path.split(csv_file)[1]
            plate_name = plate_name[:plate_name.find('_tagger')]
            self.outliers[plate_name] = self._load_state(csv_file)

    @staticmethod
    def _load_state(csv_file):
        state = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            file_states = list(reader)
            for fs in file_states:
                filename = fs['filename']
                # skip file extension
                filename = os.path.splitext(filename)[0]
                label = int(fs['label'])
                # update state
                state[filename] = label
        return state

    def is_outlier(self, plate_name, img_file):
        """
        Returns True if a given image (img_file) from a given well (well_name) is an outlier, False otherwise.
        WARN: currently we treat everything not labeled as 0 to be an outlier (i.e. images labeled as skipped -1
        are counted as outliers).
        """
        assert plate_name in self.outliers, f'Well name: {plate_name} not found. Loaded plates: {list(self.outliers.keys())}'
        plate_state = self.outliers[plate_name]

        # skip file extension if any
        img_file = os.path.splitext(img_file)[0]
        assert img_file in plate_state, f'Cannot find image file: {img_file} in plate: {plate_name}'

        label = plate_state[img_file]

        return label != 0
