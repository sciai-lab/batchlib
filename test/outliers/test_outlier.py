import glob
import os
import unittest
from pathlib import Path

from batchlib.outliers.outlier import OutlierPredicate, plate_name_from_input_folder


class TestOutliers(unittest.TestCase):
    _global_path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'misc/tagged_outliers')

    def test_outliers(self):
        plate_name = '20200406_164555_328'
        outliers = OutlierPredicate(self._global_path, plate_name)
        img_name = 'WellA01_PointA01_0008_ChannelDAPI,WF_GFP,TRITC_Seq0008'
        assert outliers(img_name)

    def test_outlier_number(self):
        plate_files = [
            os.path.split(csv_file)[1] for csv_file in glob.glob(os.path.join(self._global_path, '*.csv'))
        ]
        plate_names = [pn[:pn.find('_tagger')] for pn in plate_files]

        plate_count = {}
        total_outlier_count = 0
        total_count = 0

        for plate_name in plate_names:
            op = OutlierPredicate(self._global_path, plate_name)
            count = 0
            outlier_count = 0
            for img_file in op.outlier_tags:
                if op(img_file):
                    outlier_count += 1
                    total_outlier_count += 1
                count += 1
                total_count += 1

            plate_count[plate_name] = outlier_count / count

        for k, v in plate_count.items():
            print(f'{k}: {v}')
        print(f'\nTotal outlier count: {total_outlier_count / total_count}')

    def test_plate_name_from_input_folder(self):
        input_folder = '/home/covid19/data/20200410_145132_254'

        plate_name = plate_name_from_input_folder(input_folder)

        assert plate_name is not None
