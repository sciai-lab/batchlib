#! /home/covid19/software/miniconda3/envs/antibodies/bin/python

import argparse
import os
from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import IlastikPrediction


def run_pixel_analysis(input_folder, folder, n_jobs, reorder):

    input_folder = os.path.abspath(input_folder)
    if folder is None:
        folder = input_folder.replace('covid-data-vibor', 'data-processed')

    ilastik_bin = '/home/covid19/software/ilastik-1.4.0b1-Linux/run_ilastik.sh'
    ilastik_project = '/home/covid19/antibodies-nuclei/ilastik/local_infection.ilp'

    n_threads_il = 8 if n_jobs == 1 else 4

    in_key = 'raw'
    out_key = 'local_infection'

    # TODO add analysis job
    job_dict = {
        Preprocess: {'run': {'reorder': reorder, 'n_jobs': n_jobs}},
        IlastikPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                      'ilastik_project': ilastik_project,
                                       'input_key': in_key,
                                       'output_key': out_key},
                            'run': {'n_jobs': n_jobs, 'n_threads': n_threads_il}}
    }

    name = 'PixelAnalysisworkflow'
    run_workflow(name, folder, job_dict, input_folder=input_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pixel analysis workflow')
    parser.add_argument('input_folder', type=str, help='')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, help='', default=1)
    parser.add_argument('--reorder', type=int, default=1, help='')

    args = parser.parse_args()

    run_pixel_analysis(args.input_folder, args.folder, args.n_jobs, bool(args.reorder))
