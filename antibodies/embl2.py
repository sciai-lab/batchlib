#! /home/covid19/software/miniconda3/envs/antibodies/bin/python

import argparse
import os
import time
import h5py

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D


def run_instance_analysis(input_folder, folder, n_jobs, reorder, gpu_id, force_recompute):

    input_folder = os.path.abspath(input_folder)
    if folder is None:
        folder = input_folder.replace('covid-data-vibor', 'data-processed-test')

    model_root = '/g/kreshuk/pape/Work/covid/antibodies-nuclei/stardist/models/pretrained'
    model_name = '2D_dsb2018'

    barrel_corrector_path = '/g/kreshuk/pape/Work/covid/antibodies-nuclei/barrel_correction/barrel_corrector.h5'
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = f['data'][:]

    in_key = 'raw'
    bd_key = 'pmap_tritc'
    mask_key = 'mask'
    nuc_key = 'nucleus_segmentation'
    seg_key = 'cell_segmentation'

    torch_model_path = os.path.join('/g/kreshuk/pape/Work/covid/antibodies-nuclei',
                                    'unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }
    batch_size = 4

    # TODO add analysis job
    job_dict = {
        Preprocess: {'run': {'reorder': reorder,
                             'n_jobs': n_jobs,
                             'barrel_corrector': barrel_corrector}},
        TorchPrediction: {'build': {'input_key': in_key,
                                    'output_key': [mask_key, bd_key],
                                    'model_path': torch_model_path,
                                    'model_class': torch_model_class,
                                    'model_kwargs': torch_model_kwargs,
                                    'input_channel': 2},
                          'run': {'gpu_id': gpu_id,
                                  'batch_size': batch_size,
                                  'threshold_channels': {0: 0.5}}},
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
    run_workflow(name, folder, job_dict, input_folder=input_folder, force_recompute=force_recompute)
    t0 = time.time() - t0
    print("Run analysis pipeline in", t0, "s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run instance analysis workflow')
    parser.add_argument('input_folder', type=str, help='')
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, help='', default=1)
    parser.add_argument('--reorder', type=int, default=1, help='')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--force_recompute', type=int, default=0)

    args = parser.parse_args()
    run_instance_analysis(args.input_folder, args.folder, args.n_jobs,
                          bool(args.reorder), args.gpu_id, bool(args.force_recompute))
