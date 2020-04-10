#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import argparse
import os
import time

import h5py

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import IlastikPrediction


# TODO all kwargs should go into config file
# NOTE ignore_nvalid_inputs / ignore_failed_outputs are not used yet in the function but will be eventually
def run_pixel_analysis1(input_folder, folder, n_cpus,
                        root='/home/covid19/antibodies-nuclei', output_root_name='data-processed',
                        use_unique_output_folder=False,
                        force_recompute=False, ignore_invalid_inputs=None, ignore_failed_outputs=None):
    name = 'PixelAnalysisWorkflow1'

    input_folder = os.path.abspath(input_folder)
    if folder is None:
        folder = input_folder.replace('covid-data-vibor', output_root_name)
        if use_unique_output_folder:
            folder += '_' + name

    ilastik_bin = os.path.join(root, 'ilastik/run_ilastik.sh')
    ilastik_project = os.path.join(root, 'ilastik/local_infection.ilp')

    n_threads_il = None if n_cpus == 1 else 4

    # TODO these should also come from the config!
    in_key = 'raw'
    out_key = 'local_infection'

    barrel_corrector_path = os.path.join(root, 'barrel_correction/barrel_corrector.h5')
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = (f['divisor'][:], f['offset'][:])

    # TODO add analysis job
    job_dict = {
        Preprocess: {'run': {'n_jobs': n_cpus,
                             'barrel_corrector': barrel_corrector}},
        IlastikPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                      'ilastik_project': ilastik_project,
                                      'input_key': in_key,
                                      'output_key': out_key},
                            'run': {'n_jobs': n_cpus, 'n_threads': n_threads_il}}
    }

    t0 = time.time()
    run_workflow(name, folder, job_dict, input_folder=input_folder)
    t0 = time.time() - t0
    print("Run", name, "in", t0, "s")
    return name, t0


if __name__ == '__main__':
    doc = """Run pixel analysis workflow
    Based on ilastik preidctions.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('input_folder', type=str, help='folder with input files as tifs')
    parser.add_argument('n_cpus', type=int, help='number of cpus')
    parser.add_argument('--folder', type=str, default=None, help=fhelp)

    args = parser.parse_args()
    run_pixel_analysis1(args.input_folder, args.folder, args.n_cpus)
