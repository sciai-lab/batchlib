#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python
from batchlib.workflows import run_cell_analysis, cell_analysis_parser

parser = cell_analysis_parser('./configs', 'test_cell_analysis.conf')
config = parser.parse_args()
run_cell_analysis(config)
