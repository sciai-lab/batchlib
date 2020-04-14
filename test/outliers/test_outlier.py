import os
import unittest
from pathlib import Path

from batchlib.outliers.outlier import Outliers


class TestOutliers(unittest.TestCase):
    _global_path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'misc/tagged_outliers')

    def test_outliers(self):
        outliers = Outliers(self._global_path)
        plate_name = '20200406_164555_328'
        img_name = 'WellA01_PointA01_0008_ChannelDAPI,WF_GFP,TRITC_Seq0008'
        assert outliers.is_outlier(plate_name, img_name)
