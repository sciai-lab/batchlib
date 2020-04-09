#! /home/covid19/software/miniconda3/envs/antibodies/bin/python

import argparse
import os
import time

import h5py

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import BoundaryAndMaskPrediction, SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction


def run_instance_analysis(input_folder, folder, n_jobs, reorder, gpu_id):

    input_folder = os.path.abspath(input_folder)
    if folder is None:
        folder = input_folder.replace('covid-data-vibor', 'data-processed')

    ilastik_bin = '/home/covid19/software/ilastik-1.4.0b1-Linux/run_ilastik.sh'
    ilastik_project = '/home/covid19/antibodies-nuclei/ilastik/boundaries_and_foreground.ilp'

    model_root = '/home/covid19/antibodies-nuclei/stardist/models/pretrained'
    model_name = '2D_dsb2018'

    barrel_corrector_path = '/home/covid19/antibodies-nuclei/barrel_correction/barrel_corrector.h5'
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = f['data'][:]

    in_key = 'raw'
    bd_key = 'pmap_tritc'
    mask_key = 'mask'
    nuc_key = 'nucleus_segmentation'
    seg_key = 'cell_segmentation'

    n_threads_il = 8 if n_jobs == 1 else 4

    # TODO add analysis job
    job_dict = {
        Preprocess: {'run': {'reorder': reorder,
                             'n_jobs': n_jobs,
                             'barrel_corrector': barrel_corrector}},
        BoundaryAndMaskPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                              'ilastik_project': ilastik_project,
                                              'input_key': in_key,
                                              'boundary_key': bd_key,
                                              'mask_key': mask_key},
                                    'run': {'n_jobs': n_jobs, 'n_threads': n_threads_il}},
        StardistPrediction: {'build': {'model_root': model_root,
                                       'model_name': model_name,
                                       'input_key': in_key,
                                       'output_key': nuc_key,
                                       'input_channel': 0},
                             'run': {'gpu_id': gpu_id}},
        SeededWatershed: {'build': {'pmap_key': bd_key,
                                    'seed_key': nuc_key,
                                    'output_key': seg_key,
                                    'mask_key': mask_key},
                          'run': {'erode_mask': 3,
                                  'dilate_seeds': 3,
                                  'n_jobs': n_jobs}}
    }

    name = 'InstanceAnalysisWorkflow'
    t0 = time.time()
    run_workflow(name, folder, job_dict, input_folder=input_folder)
    t0 = time.time() - t0
    print("Run", name, "in", t0, "s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run instance analysis workflow')
    parser.add_argument('input_folder', type=str, help='')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, help='', default=1)
    parser.add_argument('--reorder', type=int, default=1, help='')
    parser.add_argument('--gpu_id', type=int, default=None)

    args = parser.parse_args()
    run_instance_analysis(args.input_folder, args.folder, args.n_jobs, bool(args.reorder), args.gpu_id)
