import os
from glob import glob

import h5py
from batchlib.reporting.export_tables import export_cell_tables


# TODO also export image and well tables
def export_single_plate(folder):

    plate_name = os.path.split(folder)[1]

    path = glob(os.path.join(folder, '*.h5'))[0]
    table_roots = [
        'tables/cell_segmentation_mean',
        'tables/cell_segmentation_sum'
    ]
    table_names = []
    with h5py.File(path, 'r') as f:
        for root in table_roots:
            tables = list(f[root].keys())
            root_name = root.split('/')[-1]
            tables = [root_name + '/' + tab for tab in tables]
            table_names.extend(tables)

    for table_name in table_names:
        identifier, channel_name = table_name.split('/')
        identifier = identifier.replace('segmentation', 'table')
        output_path = os.path.join(folder, f'{plate_name}_{identifier}_{channel_name}.csv')
        export_cell_tables(folder, output_path, table_name, n_jobs=32)


if __name__ == '__main__':
    folder = '/g/kreshuk/data/covid/sandbox/manuscript/20200417_152052_943'
    export_single_plate(folder)
