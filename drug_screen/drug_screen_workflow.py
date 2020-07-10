#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python
from batchlib.workflows.drug_screen import run_drug_screen_analysis, drug_screen_parser

parser = drug_screen_parser('./configs', 'drug_screen.conf')
config = parser.parse_args()
run_drug_screen_analysis(config)
