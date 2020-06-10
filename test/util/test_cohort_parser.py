import unittest

from batchlib.util.cohort_parser import CohortIdParser, get_cohort


class TestCohortIdParser(unittest.TestCase):
    def test_cohort_parsing(self):
        cohort_id_parser = CohortIdParser()

        # get cohort ids for some random plates/wells
        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('20200406_210102_953')
        assert plate_cohorts['C05'] == 'C50+55f'
        assert plate_cohorts['H12'] == 'B85'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('20200420_164920_764')
        assert plate_cohorts['A02'] == 'B86'
        assert plate_cohorts['H04'] == 'A39'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plateT3rep1_20200509_152617_891')
        assert plate_cohorts['C04'] == '3-0099 E'
        assert plate_cohorts['F10'] == '3-0124 K'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plateK26rep1_20200515_221809_658')
        assert plate_cohorts['A02'] == 'K1323'
        assert plate_cohorts['G05'] == 'K1353'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plate9_3rep1_20200513_160853_327')
        assert plate_cohorts['A02'] == 'F1b'
        assert plate_cohorts['G05'] == 'P181'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plate9_2rep2_20200515_230124_149')
        assert plate_cohorts['A02'] == 'P126'
        assert plate_cohorts['G05'] == 'P151'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plateU13_T9rep1_20200516_105403_122')
        assert plate_cohorts['A02'] == '02-0486-V'
        assert plate_cohorts['G04'] == '02-0497-M'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plate9_4rep1_20200604_175423_514')
        assert plate_cohorts['D11'] == 'CMV 7'
        assert plate_cohorts['G08'] == 'C4k'

        plate_cohorts = cohort_id_parser.get_cohorts_for_plate('plate9_5rep1_20200604_225512_896')
        assert plate_cohorts['A08'] == 'C209'
        assert plate_cohorts['H06'] == 'EBV 59'

    def test_cohort_extractor(self):
        assert get_cohort('CMV 20') == 'E'
        assert get_cohort('EBV 20') == 'E'
        assert get_cohort('C4k') == 'C'
        assert get_cohort('2-2100 K') == 'K'
        assert get_cohort('3-2100 K') == 'K'
        assert get_cohort('02-2100 V') == 'V?'