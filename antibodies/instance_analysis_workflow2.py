#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import os
import time

import configargparse

from batchlib import run_workflow
from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
from batchlib.preprocessing import Preprocess
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.util.logging import get_logger

logger = get_logger('Workflow.InstanceAnalysis2')


def get_input_keys(config):

    nuc_in_key = 'nuclei'
    serum_in_key = 'serum'
    marker_in_key = 'marker'

    if config.segmentation_on_corrected:
        nuc_seg_in_key = nuc_in_key + '_corrected'
        serum_seg_in_key = serum_in_key + '_corrected'
    else:
        nuc_seg_in_key = nuc_in_key
        serum_seg_in_key = serum_in_key

    if config.analysis_on_corrected:
        serum_ana_in_key = serum_in_key + '_corrected'
        marker_ana_in_key = marker_in_key + '_corrected'
    else:
        serum_ana_in_key = serum_in_key
        marker_ana_in_key = marker_in_key

    return nuc_seg_in_key, serum_seg_in_key, marker_ana_in_key, serum_ana_in_key


def run_instance_analysis2(config):
    name = 'InstanceAnalysisWorkflow2'

    # to allow running on the cpu
    if config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    model_root = os.path.join(config.root, 'stardist/models/pretrained')
    model_name = '2D_dsb2018'

    barrel_corrector_path = os.path.join(os.path.split(__file__)[0], '../misc/', config.barrel_corrector)

    torch_model_path = os.path.join(config.root,
                                    'unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    # get keys and identifier
    (nuc_seg_in_key, serum_seg_in_key,
     marker_ana_in_key, serum_ana_in_key) = get_input_keys(config)
    analysis_folder = 'instancewise_analysis_corrected' if config.analysis_on_corrected else 'instancewise_analysis'

    job_dict = {
        Preprocess.from_folder: {'build': {'input_folder': config.input_folder,
                                           'barrel_corrector_path': barrel_corrector_path,
                                           'scale_factors': config.scale_factors},
                                 'run': {'n_jobs': config.n_cpus}},
        TorchPrediction: {'build': {'input_key': serum_seg_in_key,
                                    'output_key': [config.mask_key, config.bd_key],
                                    'model_path': torch_model_path,
                                    'model_class': torch_model_class,
                                    'model_kwargs': torch_model_kwargs,
                                    'scale_factors': config.scale_factors},
                          'run': {'gpu_id': config.gpu,
                                  'batch_size': config.batch_size,
                                  'threshold_channels': {0: 0.5}}},
        StardistPrediction: {'build': {'model_root': model_root,
                                       'model_name': model_name,
                                       'input_key': nuc_seg_in_key,
                                       'output_key': config.nuc_key,
                                       'scale_factors': config.scale_factors},
                             'run': {'gpu_id': config.gpu if not config.stardist_on_cpu else None,
                                     'n_jobs': config.n_cpus}},
        SeededWatershed: {'build': {'pmap_key': config.bd_key,
                                    'seed_key': config.nuc_key,
                                    'output_key': config.seg_key,
                                    'mask_key': config.mask_key,
                                    'scale_factors': config.scale_factors},
                          'run': {'erode_mask': 20,
                                  'dilate_seeds': 3,
                                  'n_jobs': config.n_cpus}},
        CellLevelAnalysis: {'build': {'serum_key': serum_ana_in_key,
                                      'marker_key': marker_ana_in_key,
                                      'nuc_seg_key': config.nuc_key,
                                      'cell_seg_key': config.seg_key,
                                      'output_folder': analysis_folder},
                            'run': {'gpu_id': config.gpu}}
    }

    if config.skip_analysis:
        job_dict.pop(CellLevelAnalysis)

    t0 = time.time()
    run_workflow(name,
                 config.folder,
                 job_dict,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)
    t0 = time.time() - t0
    logger.info(f"Run {name} in {t0}s")
    return name, t0


def parse_instance_config2():
    return parser().parse_args()


def parser():
    doc = """Run instance analysis workflow
    Based on ilastik pixel prediction, stardist nucleus prediction
    and watershed segmentation.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    default_config = os.path.join(os.path.split(__file__)[0], 'configs', 'instance_analysis_2.conf')
    parser = configargparse.ArgumentParser(description=doc,
                                           default_config_files=[default_config],
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)

    # mandatory
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--gpu', required=True, type=int, help='id of gpu for this job')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default="", help=fhelp)

    # barrel corrector
    parser.add('--barrel_corrector', type=str, default="barrel_corrector.h5",
               help="name of barrel corrector file in ../misc/")

    # as tensorflow / pytorch gpu issue workaround
    parser.add('--stardist_on_cpu', default=False, action='store_true')

    # folder options
    parser.add("--root", default='/home/covid19/antibodies-nuclei')
    parser.add("--output_root_name", default='data-processed')
    parser.add("--use_unique_output_folder", default=False)

    # keys for intermediate data
    parser.add("--bd_key", default='boundaries', type=str)
    parser.add("--mask_key", default='mask', type=str)
    parser.add("--nuc_key", default='nucleus_segmentation', type=str)
    parser.add("--seg_key", default='cell_segmentation', type=str)

    # whether to skip the final analysis (this is to circumvent a bug regarding tensorflow not freeing the memory)
    parser.add_argument('--skip_analysis', dest='skip_analysis', default=False, action='store_true')

    # whether to run the segmentation / analysis on the corrected or on the corrected data
    parser.add("--segmentation_on_corrected", default=True)
    parser.add("--analysis_on_corrected", default=True)

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)

    default_scale_factors = None
    # default_scale_factors = [1, 2, 4, 8]
    parser.add("--scale_factors", default=default_scale_factors)

    return parser


if __name__ == '__main__':
    config = parse_instance_config2()
    run_instance_analysis2(config)
