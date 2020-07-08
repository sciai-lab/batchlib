#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python
import json
from batchlib.workflows.drug_screen import run_drug_screen_analysis, drug_screen_parser
# TODO implement running the plates in parallel

is_wfs = []
with open('./wf_plates.json') as f:
    wf_plates = json.load(f)
is_wfs += len(wf_plates) * [1]

with open('./conf_plates.json') as f:
    conf_plates = json.load(f)
is_wfs += len(wf_plates) * [0]

plates = wf_plates + conf_plates

print("Running", len(plates), "plates")


for plate, is_wf in zip(plates, is_wfs):
    parser = drug_screen_parser('./configs', 'drug_screen.conf')
    config = parser.parse_args()
    config.input_folder = plate
    config.is_wf = is_wf
    run_drug_screen_analysis(config)
