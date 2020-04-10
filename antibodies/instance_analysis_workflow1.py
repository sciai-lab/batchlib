#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import argparse
import os
import time

import h5py

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import BoundaryAndMaskPrediction, SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.analysis.cell_level_analysis import CellLevelAnalysis

# Make it easier to change between roots (which is not a good name for this ...)
ROOT = '/home/covid19/antibodies-nuclei'
# ROOT = '/home/pape/Work/covid/antibodies-nuclei'


# TODO all kwargs should go into config file
# NOTE ignore_nvalid_inputs / ignore_failed_outputs are not used yet in the function but will be eventually
def run_instance_analysis1(input_folder, folder, gpu, n_cpus,
                           root=ROOT, output_root_name='data-processed', use_unique_output_folder=False,
                           force_recompute=False, ignore_invalid_inputs=None, ignore_failed_outputs=None):
    name = 'InstanceAnalysisWorkflow1'

    # to allow running on the cpu
    if gpu < 0:
        gpu = None

    input_folder = os.path.abspath(input_folder)
    if folder is None:
        folder = input_folder.replace('covid-data-vibor', output_root_name)
        if use_unique_output_folder:
            folder += '_' + name

    ilastik_bin = os.path.join(root, 'ilastik/run_ilastik.sh')
    ilastik_project = os.path.join(root, 'ilastik/boundaries_and_foreground.ilp')

    model_root = os.path.join(root, 'stardist/models/pretrained')
    model_name = '2D_dsb2018'

    barrel_corrector_path = os.path.join(root, 'barrel_correction/barrel_corrector.h5')
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = f['data'][:]

    # TODO these should also come from the config!
    in_key = 'raw'
    bd_key = 'pmap_tritc'
    mask_key = 'mask'
    nuc_key = 'nucleus_segmentation'
    seg_key = 'cell_segmentation'

    n_threads_il = None if n_cpus == 1 else 4

    job_dict = {
        Preprocess: {'run': {'n_jobs': n_cpus,
                             'barrel_corrector': barrel_corrector}},
        BoundaryAndMaskPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                              'ilastik_project': ilastik_project,
                                              'input_key': in_key,
                                              'boundary_key': bd_key,
                                              'mask_key': mask_key},
                                    'run': {'n_jobs': n_cpus,
                                            'n_threads': n_threads_il}},
        StardistPrediction: {'build': {'model_root': model_root,
                                       'model_name': model_name,
                                       'input_key': in_key,
                                       'output_key': nuc_key,
                                       'input_channel': 0},
                             'run': {'gpu_id': gpu}},
        SeededWatershed: {'build': {'pmap_key': bd_key,
                                    'seed_key': nuc_key,
                                    'output_key': seg_key,
                                    'mask_key': mask_key},
                          'run': {'erode_mask': 3,
                                  'dilate_seeds': 3,
                                  'n_jobs': n_cpus}},
        CellLevelAnalysis: {'build': {'raw_key': in_key,
                                      'nuc_seg_key': nuc_key,
                                      'cell_seg_key': seg_key},
                            'run': {'gpu_id': gpu}}
    }

    t0 = time.time()
    run_workflow(name, folder, job_dict, input_folder=input_folder)
    t0 = time.time() - t0
    print("Run", name, "in", t0, "s")
    return name, t0


if __name__ == '__main__':
    doc = """Run instance analysis workflow
    Based on torch prediction, stardist nucleus prediction
    and watershed segmentation.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('input_folder', type=str, help='folder with input files as tifs')
    parser.add_argument('gpu', type=int, help='id of gpu for this job')
    parser.add_argument('n_cpus', type=int, help='number of cpus')
    parser.add_argument('--folder', type=str, default=None, help=fhelp)

    args = parser.parse_args()
    run_instance_analysis1(args.input_folder, args.folder, args.gpu, args.n_cpus)
