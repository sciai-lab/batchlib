import csv
import glob
import os

from batchlib.util.logging import get_logger

logger = get_logger('Workflow.Outliers')

DEFAULT_OUTLIER_DIR = os.path.join(os.path.split(__file__)[0], '../../misc/tagged_outliers')


def get_outlier_predicate(config):
    if hasattr(config, 'outliers_dir') and config.outliers_dir is not None:
        outliers_dir = config.outliers_dir
    else:
        outliers_dir = DEFAULT_OUTLIER_DIR

    if hasattr(config, 'plate_name') and config.plate_name is not None:
        plate_name = config.plate_name
        if isinstance(plate_name, list):
            assert len(plate_name) == 1, \
                f'Only a single plate name is allowed, but {len(plate_name)} were given to the outlier predicate'
            plate_name = plate_name[0]
        logger.info(f'Using plate name: {plate_name}')
    else:
        logger.info(f"Trying to parse 'plate_name' from the input folder: {config.input_folder}")
        plate_name = plate_name_from_input_folder(config.input_folder)
        if plate_name is not None:
            logger.info(f"plate_name found: {plate_name}")
        else:
            logger.info(f"Cannot parse plate_name. Outlier detection will be skipped")
            # no plate name was given and it cannot be parsed from the config.input_folder
            # so we skip outlier detection, i.e. assign outlier: None to each of the images
            return lambda im: None

    return OutlierPredicate(root_table_dir=outliers_dir, plate_name=plate_name)


def plate_name_from_input_folder(input_folder):
    for csv_file in glob.glob(os.path.join(DEFAULT_OUTLIER_DIR, '*.csv')):
        plate_name = os.path.split(csv_file)[1]
        plate_name = plate_name[:plate_name.find('_tagger')]

        if plate_name in input_folder:
            return plate_name
    return None


class OutlierPredicate:
    def __init__(self, root_table_dir, plate_name):
        """
        Iterates over the CSV files stored in 'root_table_dir', where each CSV stores outlier tagging to a given plate,
        finds a CSV corresponding to a given 'plate_name' and loads the outlier tag for each image in the plate.
        Labels are:
            0 - accepted image
            1 - outlier image
            -1 - skipped (not tagged) image

        If a given 'plate_name' cannot be found in the 'root_table_dir' a warning is raised.
        """

        assert plate_name is not None, 'plate_name cannot be None'

        self.outlier_tags = None

        # parse outliers
        for csv_file in glob.glob(os.path.join(root_table_dir, '*.csv')):
            assert '_tagger_state.csv' in csv_file
            pn = os.path.split(csv_file)[1]
            pn = pn[:pn.find('_tagger')]
            if pn == plate_name:
                self.outlier_tags = self._load_state(csv_file)

        if self.outlier_tags is None:
            # plate_name could not be found in root_table_dir
            logger.warning(f'Plate name: {plate_name} could not be found in {root_table_dir}. '
                           f'Outlier detection will be skipped. Make sure to put the outlier CSV file inside the {root_table_dir}.')

    def __call__(self, img_file):
        """
        Check if a given image (img_file) was marked as an outlier.

        Returns:
            True if the img_file was marked as an outlier
            False if the img_file was marked as a valid image
            None if the img_file was skipped during tagging, or outliers are not available for a given plate (no outlier CSV file)
        """

        # take only the file
        img_file = os.path.split(img_file)[1]
        # skip file extension if any
        img_file = os.path.splitext(img_file)[0]

        if self.outlier_tags is None:
            # outliers info not available
            return None

        label = self.outlier_tags[img_file]
        if label == 1:
            return True
        elif label == 0:
            return False
        elif label == -1:
            return None
        else:
            raise ValueError(f'Unsupported outlier label value: {label}')

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
