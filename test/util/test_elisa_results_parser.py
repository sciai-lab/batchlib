import unittest
import math

from batchlib.util.elisa_results_parser import ElisaResultsParser


class TestElisaResultsParser(unittest.TestCase):
    def test_elisa_results_parsing(self):
        elisa_results_parser = ElisaResultsParser()

        results = elisa_results_parser.get_elisa_values('C14d', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 0.96
        assert results[1] == 0.39

        results = elisa_results_parser.get_elisa_values('C23i', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 6.36
        assert results[1] == 6.16

        results = elisa_results_parser.get_elisa_values('Z24', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 0.05
        assert results[1] is None

        results = elisa_results_parser.get_elisa_values('A9', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 0.14
        assert results[1] == 0.12

        results = elisa_results_parser.get_elisa_values('P12', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 7.81
        assert results[1] == 11

        results = elisa_results_parser.get_elisa_values('3-0010 K', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 0.24
        assert results[1] is None

        results = elisa_results_parser.get_elisa_values('K97', test_names=['IF IgG', 'IF IgA', 'Roche', 'Abbot', 'Luminex'])
        assert results[0] == 1.07
        assert results[1] == 1.13
        assert results[2] == 'pos'
        assert results[3] == 'neg'
        assert results[4] == 'neg'

        results = elisa_results_parser.get_elisa_values('P21', test_names=['ELISA IgG', 'ELISA IgA'])
        assert results[0] == 2.14
        assert results[1] == 2.08

        results = elisa_results_parser.get_elisa_values('C2b', test_names=['ELISA IgG', 'ELISA IgA', 'days_after_onset'])
        assert results[0] == 1.68
        assert results[1] == 4.36
        assert results[2] == 12
