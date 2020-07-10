import json
import os
from subprocess import run

plates = []
with open('./wf_plates.json') as f:
    plates += json.load(f)
with open('./conf_plates.json') as f:
    plates += json.load(f)

for plate in plates:
    name = os.path.split(plate)[1]
    table = os.path.join(plate.replace('covid-data-vibor', 'data-processed'), f'{name}_cells_table.xlsx')
    cmd = ['scp', f'pape@gpu6.cluster.embl.de:{table}', '.']
    run(cmd)
