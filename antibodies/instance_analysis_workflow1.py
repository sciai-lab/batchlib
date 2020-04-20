#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import os
import time

import configargparse

from batchlib import run_workflow
from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import BoundaryAndMaskPrediction, SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.util.logging import get_logger

logger = get_logger('Workflow.InstanceAnalysis1')


def run_instance_analysis1(config):
    name = 'InstanceAnalysisWorkflow1'
    # TODO fix for new data layout
    raise NotImplementedError("Not adapted to new data layout")

    # to allow running on the cpu
    if config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    ilastik_bin = os.path.join(config.root, 'ilastik/run_ilastik.sh')
    ilastik_project = os.path.join(config.root, 'ilastik/boundaries_and_foreground.ilp')

    model_root = os.path.join(os.path.split(__file__[0]), '../misc/models/stardist')
    model_name = '2D_dsb2018'

    barrel_corrector_path = os.path.join(os.path.split(__file__)[0],
                                         '../misc/barrel_corrector.h5')

    n_threads_il = None if config.n_cpus == 1 else 4

    job_dict = {
        Preprocess.from_folder: {'build': {'input_folder': config.input_folder,
                                           'barrel_corrector_path': barrel_corrector_path},
                                 'run': {'n_jobs': config.n_cpus}},
        BoundaryAndMaskPrediction: {'build': {'ilastik_bin': ilastik_bin,
                                              'ilastik_project': ilastik_project,
                                              'input_key': config.in_key,
                                              'boundary_key': config.bd_key,
                                              'mask_key': config.mask_key},
                                    'run': {'n_jobs': config.n_cpus,
                                            'n_threads': n_threads_il}},
        StardistPrediction: {'build': {'model_root': model_root,
                                       'model_name': model_name,
                                       'input_key': config.in_key,
                                       'output_key': config.nuc_key,
                                       'input_channel': 0},
                             'run': {'gpu_id': config.gpu,
                                     'n_jobs': config.n_cpus}},
        SeededWatershed: {'build': {'pmap_key': config.bd_key,
                                    'seed_key': config.nuc_key,
                                    'output_key': config.seg_key,
                                    'mask_key': config.mask_key},
                          'run': {'erode_mask': 3,
                                  'dilate_seeds': 3,
                                  'n_jobs': config.n_cpus}},
        CellLevelAnalysis: {'build': {'raw_key': config.in_key,
                                      'nuc_seg_key': config.nuc_key,
                                      'cell_seg_key': config.seg_key},
                            'run': {'gpu_id': config.gpu}}
    }

    t0 = time.time()
    run_workflow(name, config.folder, job_dict,
                 input_folder=config.input_folder,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)
    t0 = time.time() - t0
    logger.info(f"Run {name} in {t0}s")
    return name, t0


def parse_instance_config1():
    doc = """Run instance analysis workflow
    Based on torch prediction, stardist nucleus prediction
    and watershed segmentation.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    default_config = os.path.join(os.path.split(__file__)[0], 'configs', 'instance_analysis_1.conf')
    parser = configargparse.ArgumentParser(description=doc,
                                           default_config_files=[default_config],
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--gpu', required=True, type=int, help='id of gpu for this job')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default=None, help=fhelp)

    parser.add('--root', default='/home/covid19/antibodies-nuclei')
    parser.add('--output_root_name', default='data-processed-new')
    parser.add('--use_unique_output_folder', default=False)
    parser.add('--force_recompute', default=False)
    parser.add('--ignore_invalid_inputs', default=None)
    parser.add('--ignore_failed_outputs', default=None)

    parser.add('--in_key', default='raw', type=str)
    parser.add('--bd_key', default='pmap_tritc_ilastik', type=str)
    parser.add('--mask_key', default='mask_ilastik', type=str)
    parser.add('--nuc_key', default='nucleus_segmentation', type=str)
    parser.add('--seg_key', default='cell_segmentation_ilastik', type=str)

    # TODO add default scale factors
    default_scale_factors = None
    # default_scale_factors = [1, 2, 4, 8]
    parser.add("--scale_factors", default=default_scale_factors)

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_instance_config1()
    run_instance_analysis1(config)
