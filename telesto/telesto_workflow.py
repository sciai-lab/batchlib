#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python
from batchlib.workflows.telesto import run_telesto_analysis, telesto_parser

parser = telesto_parser('./configs', 'telesto.conf')
config = parser.parse_args()
run_telesto_analysis(config)
