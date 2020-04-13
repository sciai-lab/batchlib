from glob import glob
from run_all import run_all


def all_plates():
    in_folder = '/home/covid19/data/covid-data-vibor'
    folders = glob(in_folder + '/*')

    for folder in folders:
        run_all(folder, 0, 12)


all_plates()
