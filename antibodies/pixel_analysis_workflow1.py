#! /bin/usr/python
# TODO proper shebang

import argparse
import os
from batchlib import run_workflow
from batchlib.preprocessing import Preprocess


def run_pixel_analysis(input_folder, n_jobs, reorder):

    input_folder = os.path.abspath(input_folder)
    folder = input_folder.replace('covid-data-vibor', 'data-processed')

    job_dict = {Preprocess: {'run': {'reorder': reorder, 'n_jobs': n_jobs}}}

    run_workflow(name, folder, job_dict, input_folder=in_folder)


if __name__ == '__name__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_folder', type=str, help='')
    parser.add_argument('--n_jobs', type=int, help='')
    parser.add_argument('--reorder', type=int, default=0, help='')

    args = parser.parse_args()

    run_pixel_analysis(args.input_folder, args.n_jobs, bool(args.reorder))
