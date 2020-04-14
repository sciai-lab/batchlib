#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

from glob import glob
from run_all_workflows import run_all_workflows


def run_all_plates(with_corrected=True):
    in_folder = '/home/covid19/data/covid-data-vibor'
    folders = glob(in_folder + '/*')

    for folder in folders:
        run_all_workflows(folder)


run_all_plates()
