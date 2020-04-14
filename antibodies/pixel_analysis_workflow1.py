#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import configargparse
import os
import time
from glob import glob

import h5py

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess, get_channel_settings
from batchlib.segmentation import IlastikPrediction
from batchlib.analysis.pixel_level_analysis import PixellevelAnalysis, PixellevelPlots


def run_pixel_analysis1(config):
    name = 'PixelAnalysisWorkflow1'

    input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    ilastik_bin = os.path.join(config.root, 'ilastik/run_ilastik.sh')
    ilastik_project = os.path.join(config.root, 'ilastik/local_infection.ilp')

    n_threads_il = None if config.n_cpus == 1 else 4

    barrel_corrector_path = os.path.join(config.root, 'barrel_correction/barrel_corrector.h5')
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = (f['divisor'][:], f['offset'][:])

    # get the correct channel ordering and names for this data
    fname = glob(os.path.join(config.input_folder, '*.tiff'))[0]
    channel_names, settings, reorder = get_channel_settings(fname)

    job_dict = {
        Preprocess: {'build': {'channel_names': channel_names,
                               'viewer_settings': settings},
                     'run': {'n_jobs': config.n_cpus,
                             'barrel_corrector': barrel_corrector,
                             'reorder': reorder}},
        IlastikPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                      'ilastik_project': ilastik_project,
                                      'input_key': config.in_key,
                                      'output_key': config.out_key},
                            'run': {'n_jobs': config.n_cpus, 'n_threads': n_threads_il}},
        PixellevelAnalysis: {'build': {'raw_key': config.in_analysis_key,
                                       'infection_key': config.out_key,
                                       'output_folder': config.output_folder}},
    }

    t0 = time.time()

    run_workflow(name, config.folder, job_dict,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)

    job_dict2 = {PixellevelPlots: {}}
    run_workflow(name, config.folder, job_dict2,
                 input_folder=os.path.join(config.folder, config.output_folder),
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)

    t0 = time.time() - t0
    return name, t0


def parse_pixel_config1():
    doc = """Run pixel analysis workflow
    Based on ilastik preidctions.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    default_config = os.path.join(os.path.split(__file__)[0], 'configs', 'pixelwise_analysis.conf')
    parser = configargparse.ArgumentParser(description=doc,
                                           default_config_files=[default_config],
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)

    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default=None, help=fhelp)

    # options
    parser.add("--in_key", default='raw')
    parser.add("--in_analysis_key", default='TRITC')
    parser.add("--out_key", default='local_infection')
    parser.add("--output_folder", default="pixelwise_analysis")
    parser.add("--root", default='/home/covid19/antibodies-nuclei', type=str)
    parser.add("--output_root_name", default='data-processed-new', type=str)
    parser.add("--use_unique_output_folder", default=False, action='store_false')

    parser.add("--force_recompute", default=False)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)

    print(parser.format_values())
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_pixel_config1()
    run_pixel_analysis1(config)
