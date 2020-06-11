import os
import h5py

from batchlib.util import read_table
from process_for_manuscript import all_manuscript_plates


def get_manuscript_plates():
    manuscript_plates = all_manuscript_plates()
    root_folder = '/g/kreshuk/data/covid/data-processed'
    return [os.path.join(root_folder, plate) for plate in manuscript_plates]


def quantify_well_or_image(folder, name):
    plate_name = os.path.split(folder)[1]
    table_path = os.path.join(folder, f'{plate_name}_table.hdf5')
    with h5py.File(table_path, 'r') as f:
        cols, tab = read_table(f, f'images/outliers')
    print(cols, tab)
    n = len(tab)
    outliers = tab[:, cols.index('is_outlier')] == 1
    n_outliers = outliers.sum()
    outlier_type = tab[:, cols.index('outlier_type')]
    print(len([otype for otype in outlier_type if 'manual: -1' in otype]), '/', n)
    return n, n_outliers


def quantify_well_or_image_outlier(name):

    plates = get_manuscript_plates()
    n_samples, n_outliers = 0, 0
    for plate in plates:
        ns, no = quantify_well_or_image(plate, name)
        n_samples += ns
        n_outliers += no

    print("Outliers per", name, ":")
    print(n_outliers, "/", n_samples, "=", float(n_outliers) / n_samples)


if __name__ == '__main__':
    quantify_well_or_image_outlier('wells')
    print()
    quantify_well_or_image_outlier('images')
