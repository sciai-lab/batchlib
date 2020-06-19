import os
from glob import glob
import pandas as pd
from batchlib.util import image_name_to_well_name

manuscript_plates = [
    "20200417_132123_311",
    "20200417_152052_943",
    "20200420_164920_764",
    "20200420_152417_316",
    "plate1_IgM_20200527_125952_707",
    "plate2_IgM_20200527_155923_897",
    "plate5_IgM_20200528_094947_410",
    "plate6_IgM_20200528_111507_585",
    "plate9_4_IgM_20200604_212451_328",
    "plate9_4rep1_20200604_175423_514",
    "plate9_5_IgM_20200605_084742_832",
    "plate9_5rep1_20200604_225512_896"
]

summary_table = '/g/kreshuk/software/batchlib/antibodies/embl/manuscript_plates_20200615/manuscript_plates_20200615_scores.xlsx'
summary_table = pd.read_excel(summary_table)
cols = summary_table.columns.values.tolist()
summary_table = summary_table.values

plate_id = cols.index('plate_name')
well_id = cols.index('well_name')
cohort_id = cols.index('cohort_id')

summary_dict = {}
for row in summary_table:
    plate_name = row[plate_id]
    well_name = row[well_id]
    cohort = row[cohort_id]
    if plate_name in summary_dict:
        summary_dict[plate_name].update({well_name: cohort})
    else:
        summary_dict[plate_name] = {well_name: cohort}

table = []
columns = ['Files', 'Plate', 'Well', 'Well Number', 'Organism', 'Gene', 'Gene symbol', 'Comments']

orga = 'VeroE6 + human serum'
gene = ''
gene_symbol = ''

root = '/g/kreshuk/data/covid/for-biostudies'

for plate in manuscript_plates:
    plate_folder = os.path.join(root, plate)
    im_files = glob(os.path.join(plate_folder, '*.h5'))

    if 'IgM' in plate:
        comment = 'Antibodies: IgG, IgM; Virus Marker: anti-dsRNA'
    else:
        comment = 'Antibodies: IgG, IgA; Virus Marker: anti-dsRNA'

    for im in im_files:
        im_name = os.path.split(im)[1]
        well = image_name_to_well_name(im_name)

        well_info = summary_dict[plate][well]

        row = [f'for-biostudies/{plate}/{im_name}', plate,
               well, well_info,
               orga, gene, gene_symbol,
               comment]
        table.append(row)

    tab_file = f'for-biostudies/{plate}/{plate}_table.hdf5'
    tab_comment = f'analysis results for wells in {plate}'
    row = [tab_file, plate, '', '', orga, gene, gene_symbol, tab_comment]
    table.append(row)

submission_table = pd.DataFrame(table, columns=columns)
submission_table.to_excel('covid-if-biostudies.xlsx')
