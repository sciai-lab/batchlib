import unittest
import math

from batchlib.util.elisa_results_parser import ElisaResultsParser


class TestElisaResultsParser(unittest.TestCase):
    def test_elisa_results_parsing(self):
        elisa_results_parser = ElisaResultsParser()

        igg, iga = elisa_results_parser.get_elisa_values('C14d')
        assert igg == 0.96
        assert iga == 0.39

        igg, iga = elisa_results_parser.get_elisa_values('C23i')
        assert igg == 6.36
        assert iga == 6.16

        igg, iga = elisa_results_parser.get_elisa_values('Z24')
        assert igg == 0.05
        assert math.isnan(iga)


        igg, iga = elisa_results_parser.get_elisa_values('A9')
        assert igg == 0.14
        assert iga == 0.12

        igg, iga = elisa_results_parser.get_elisa_values('P12')
        assert igg == 7.81
        assert iga == 11

        igg, iga = elisa_results_parser.get_elisa_values('3-0010 K')
        assert igg == 0.24
        assert math.isnan(iga)
