import os
from glob import glob

import numpy as np
from batchlib.util import open_file, read_table


def determine_igm_mad(root):
    plates = glob(os.path.join(root, '*IgM*'))
    print(plates)
    mads = []
    mad_key = 'serum_IgM_mad'
    for plate in plates:
        plate_name = os.path.split(plate)[1]
        table_path = os.path.join(plate, f'{plate_name}_table.hdf5')
        with open_file(table_path, 'r') as f:
            cols, table = read_table(f, 'plate/backgrounds_min_well')
        mad = table[:, cols.index(mad_key)]
        print(plate_name, ":", mad)
        if 'plate2' in plate_name:
            continue
        mads.append(mad)

    print()
    print("Mean IgM MAD:")
    print(np.mean(mads))


determine_igm_mad('/g/kreshuk/data/covid/_')
