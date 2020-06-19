#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import os
import time

import configargparse
from glob import glob

from batchlib import run_workflow
from batchlib.analysis import PixellevelAnalysis, all_plots
from batchlib.outliers.outlier import get_outlier_predicate
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import IlastikPrediction
from batchlib.util.logger import get_logger

logger = get_logger('Workflow.PixelAnalysis')


def get_input_keys(config):

    nuc_in_key = 'nuclei'
    serum_in_key = 'serum'
    marker_in_key = 'marker'

    if config.segmentation_on_corrected:
        nuc_seg_in_key = nuc_in_key + '_corrected'
        marker_seg_in_key = marker_in_key + '_corrected'
        serum_seg_in_key = serum_in_key + '_corrected'
    else:
        nuc_seg_in_key = nuc_in_key
        marker_seg_in_key = marker_in_key
        serum_seg_in_key = serum_in_key

    if config.analysis_on_corrected:
        serum_ana_in_key = serum_in_key + '_corrected'
    else:
        serum_ana_in_key = serum_in_key

    return nuc_seg_in_key, marker_seg_in_key, serum_seg_in_key, serum_ana_in_key


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

    # get keys and identifier
    (nuc_seg_in_key, marker_seg_in_key, serum_seg_in_key,
     serum_ana_in_key) = get_input_keys(config)
    analysis_folder = 'pixelwise_analysis_corrected' if config.analysis_on_corrected else 'pixelwise_analysis'

    barrel_corrector_path = os.path.join(os.path.split(__file__)[0], '../misc/', config.barrel_corrector)

    outlier_predicate = get_outlier_predicate(config)

    # TODO add pixel-level summary
    job_dict = {
        Preprocess.from_folder: {'build': {'input_folder': config.input_folder,
                                           'barrel_corrector_path': barrel_corrector_path,
                                           'scale_factors': config.scale_factors},
                                 'run': {'n_jobs': config.n_cpus}},
        IlastikPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                      'ilastik_project': ilastik_project,
                                      'input_key': [nuc_seg_in_key, marker_seg_in_key, serum_seg_in_key],
                                      'output_key': [config.out_key_infected,
                                                     config.out_key_not_infected],
                                      'keep_channels': [0, 1],
                                      'scale_factors': config.scale_factors},
                            'run': {'n_jobs': config.n_cpus, 'n_threads': n_threads_il}},
        PixellevelAnalysis: {'build': {'serum_key': serum_ana_in_key,
                                       'infected_key': config.out_key_infected,
                                       'not_infected_key': config.out_key_not_infected,
                                       'output_folder': analysis_folder},
                             'run': {'n_jobs': config.n_cpus}}
    }

    t0 = time.time()
    run_workflow(name, config.folder, job_dict,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)

    # run all plots on the output files
    output_folder = os.path.join(config.folder, analysis_folder)
    json_pattern = os.path.join(output_folder, "*.json")
    all_json_files = glob(json_pattern)
    all_plots(all_json_files, output_folder)

    t0 = time.time() - t0
    logger.info(f"Run {name} in {t0}s")
    return name, t0


def parse_pixel_config1():
    return parser().parse_args()


def parser():
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

    # mandatory
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--gpu', required=True, type=int, help='id of gpu for this job')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default=None, help=fhelp)

    # barrel corrector
    parser.add('--barrel_corrector', type=str, default="barrel_corrector.h5",
               help="name of barrel corrector file in ../misc/")

    # folder options
    parser.add("--root", default='/home/covid19/antibodies-nuclei', type=str)
    parser.add("--output_root_name", default='data-processed', type=str)
    parser.add("--use_unique_output_folder", default=False)

    # keys for intermediate data
    parser.add("--out_key_infected", default='local_infected')
    parser.add("--out_key_not_infected", default='local_not_infected')

    # whether to run the segmentation / analysis on the corrected or on the corrected data
    parser.add("--segmentation_on_corrected", default=True)
    parser.add("--analysis_on_corrected", default=True)

    default_scale_factors = [1, 2, 4, 8, 16]
    parser.add("--scale_factors", default=default_scale_factors)

    # runtime options
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)

    # tagged outliers from a given plate
    # if plate_name is empty we will try to infer it from the 'input_folder' name
    parser.add("--plate_name", default=None, nargs='+', type=str, help="The name of the imaged plate")
    # if outliers_dir is empty, ../misc/tagged_outliers will be used
    parser.add("--outliers_dir", default=None, type=str,
               help="Path to the directory containing CSV files with marked outliers")

    return parser


if __name__ == '__main__':
    config = parse_pixel_config1()
    run_pixel_analysis1(config)
