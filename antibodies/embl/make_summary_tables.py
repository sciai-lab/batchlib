import os
from glob import glob
from functools import partial
from concurrent import futures
from tqdm import tqdm
from batchlib.reporting.export_tables import export_tables_for_plate


_export = partial(export_tables_for_plate, cell_table_name='cell_segmentation_mean',
                  marker_name='marker_for_infected_classification')


folders = glob(os.path.join('/g/kreshuk/data/covid/data-processed/*'))
folders = [folder for folder in folders if (not 'plateT' in folder and not 'plateK' in folder)]

# print(folders[0])
# _export(folders[0])

n_jobs = 16
with futures.ProcessPoolExecutor(n_jobs) as tp:
    list(tqdm(tp.map(_export, folders), total=len(folders)))
