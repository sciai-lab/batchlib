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
        assert outliers(img_name) == 1

    def test_no_outlier_for_plate(self):
        plate_name = 'New_plate'
        outliers = OutlierPredicate(self._global_path, plate_name)
        img_name = 'WellA01_PointA01_0008_ChannelDAPI,WF_GFP,TRITC_Seq0008'
        assert outliers(img_name) == -1

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
                if op(img_file) == 1:
                    outlier_count += 1
                    total_outlier_count += 1
                count += 1
                total_count += 1

            plate_count[plate_name] = outlier_count / count

        for k, v in plate_count.items():
            print(f'{k}: {v}')
        print(f'\nTotal outlier count: {total_outlier_count / total_count}')

    def test_outlier_number_for_manuscirpt(self):
        manuscript_plates = [
            "20200417_132123_311",
            "20200417_152052_943",
            "20200417_172611_193",
            "20200417_185943_790",
            "20200420_152417_316",
            "20200420_164920_764",
            "plate1_IgM_20200527_125952_707",
            "plate2_IgM_20200527_155923_897",
            "plate5_IgM_20200528_094947_410",
            "plate6_IgM_20200528_111507_585",
            "plate7_IgM_20200602_162805_201",
            "plate7rep1_20200426_103425_693",
            "plate8_IgM_20200529_144442_538",
            "plate8rep1_20200425_162127_242",
            "plate8rep2_20200502_182438_996",
            "plate9_2rep2_20200515_230124_149",
            "plate9_3rep1_20200513_160853_327",
            "plate9rep1_20200430_144438_974"
        ]
        total_outlier_count = 0
        total_count = 0

        for plate_name in manuscript_plates:
            op = OutlierPredicate(self._global_path, plate_name)
            count = 0
            outlier_count = 0
            if op.outlier_tags is not None:
                for img_file in op.outlier_tags:
                    if op(img_file) == 1:
                        outlier_count += 1
                        total_outlier_count += 1
                    count += 1
                    total_count += 1
            else:
                print('Missing plate: ' + plate_name)

        print(f'\nTotal outlier count (manuscript plates): {total_outlier_count / total_count}')

    def test_plate_name_from_input_folder(self):
        input_folder = '/home/covid19/data/20200410_145132_254'

        plate_name = plate_name_from_input_folder(input_folder)

        assert plate_name is not None
