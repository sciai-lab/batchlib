#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python
from batchlib.workflows import mean_and_sum_cell_analysis, cell_analysis_parser

parser = cell_analysis_parser('./configs', 'cell_analysis.conf')
config = parser.parse_args()
mean_and_sum_cell_analysis(config)
