#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import os
import time
from glob import glob

import configargparse
import h5py

from batchlib import run_workflow
from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
from batchlib.preprocessing import Preprocess, get_channel_settings
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.util.logging import get_logger

logger = get_logger('Workflow.InstanceAnalysis2')


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

    barrel_corrector_path = os.path.join(config.root, 'barrel_correction/barrel_corrector.h5')
    with h5py.File(barrel_corrector_path, 'r') as f:
        barrel_corrector = (f['divisor'][:], f['offset'][:])

    torch_model_path = os.path.join(config.root,
                                    'unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    analysis_identifier = None
    if config.in_key_analysis != 'raw':
        analysis_identifier = config.in_key_analysis

    # get the correct channel ordering and names for this data
    fname = glob(os.path.join(config.input_folder, '*.tiff'))[0]
    channel_names, settings, reorder = get_channel_settings(fname)

    job_dict = {
        Preprocess: {'build': {'channel_names': channel_names,
                               'viewer_settings': settings},
                     'run': {'n_jobs': config.n_cpus,
                             'barrel_corrector': barrel_corrector,
                             'reorder': reorder}},
        TorchPrediction: {'build': {'input_key': config.in_key,
                                    'output_key': [config.mask_key, config.bd_key],
                                    'model_path': torch_model_path,
                                    'model_class': torch_model_class,
                                    'model_kwargs': torch_model_kwargs,
                                    'input_channel': 2},
                          'run': {'gpu_id': config.gpu,
                                  'batch_size': config.batch_size,
                                  'threshold_channels': {0: 0.5}}},
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
        CellLevelAnalysis: {'build': {'raw_key': config.in_key_analysis,
                                      'nuc_seg_key': config.nuc_key,
                                      'cell_seg_key': config.seg_key,
                                      'identifier': analysis_identifier},
                            'run': {'gpu_id': config.gpu}}
    }

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

    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--gpu', required=True, type=int, help='id of gpu for this job')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default="", help=fhelp)

    # options
    parser.add("--batch_size", default=4)
    parser.add("--root", default='/home/covid19/antibodies-nuclei')
    parser.add("--output_root_name", default='data-processed-new')
    parser.add("--use_unique_output_folder", default=False)
    parser.add("--force_recompute", default=False)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)

    parser.add("--in_key", default='raw', type=str)
    parser.add("--in_key_analysis", default='raw', type=str)
    parser.add("--bd_key", default='pmap_tritc', type=str)
    parser.add("--mask_key", default='mask', type=str)
    parser.add("--nuc_key", default='nucleus_segmentation', type=str)
    parser.add("--seg_key", default='cell_segmentation', type=str)

    logger.info(parser.format_values())
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_instance_config2()
    run_instance_analysis2(config)
