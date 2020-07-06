import os
import time
import configargparse

import pandas as pd

from batchlib import run_workflow
# from batchlib.analysis.cell_analysis_qc import ImageLevelQC
from batchlib.analysis.drug_screen_analysis import DrugScreenAnalysisCellTable
from batchlib.analysis.feature_extraction import InstanceFeatureExtraction
from batchlib.segmentation.segmentation_workflows import nucleus_segmentation_workflow
from batchlib.preprocessing import Preprocess, get_barrel_corrector
from batchlib.util import open_file, read_table
from batchlib.util.logger import get_logger
from batchlib.workflows.cell_analysis import get_barrel_corrector_folder

logger = get_logger('Workflow.CellAnalysis')


# manually measured by Vibor
BACKGROUNDS_WF = {
    "nuclei": 2500,
    "sensor": 1900,
    "infection marker": 600,
    "infection marker2": 880
}

BACKGROUNDS_CONF = {
    "nuclei": 900,
    "sensor": 550,
    "infection marker": 550,
    "infection marker2": 680
}


def _barrel_corrector(config):
    if config.barrel_corrector_folder == 'auto':
        barrel_corrector_folder = get_barrel_corrector_folder(config)
    else:
        barrel_corrector_folder = config.barrel_corrector_folder
    barrel_corrector_path = get_barrel_corrector(barrel_corrector_folder, config.input_folder)
    if not os.path.exists(barrel_corrector_path):
        raise ValueError(f"Invalid barrel corrector path {barrel_corrector_path}")
    return barrel_corrector_path


def export_cell_table(folder, plate_name=None):
    if plate_name is None:
        plate_name = os.path.split(folder.rstrip('/'))[1]
    in_table_path = os.path.join(folder, f'{plate_name}_table.hdf5')
    out_table_path = os.path.join(folder, f'{plate_name}_cells_table.xlsx')

    print("Read cell table from hdf5 ...")
    with open_file(in_table_path, 'r') as f:
        cols, tab = read_table(f, 'cells/default')

    df = pd.DataFrame(tab, columns=cols)
    print("Write cell table to excel ...")
    df.to_excel(out_table_path, index=False)


def core_ds_workflow_tasks(config, nuc_seg_in_key):
    semantic_viewer_settings = {'nuclei': {'color': 'Blue', 'visible': True},
                                'infection marker': {'color': 'Red', 'visible': True},
                                'infection marker2': {'color': 'Red', 'visible': False},
                                'sensor': {'color': 'Green', 'visible': False}}

    # is this from the wide-field set-up?
    is_wf = bool(config.is_wf)
    if is_wf:
        barrel_corrector_path = _barrel_corrector(config)
        backgrounds = BACKGROUNDS_WF
    else:
        barrel_corrector_path = None
        backgrounds = BACKGROUNDS_CONF

    job_list = [
        (Preprocess.from_folder, {
            'build': {
                'input_folder': config.input_folder,
                'barrel_corrector_path': barrel_corrector_path,
                'scale_factors': config.scale_factors,
                'semantic_settings': semantic_viewer_settings},
            'run': {
                'n_jobs': config.n_cpus}})
    ]

    # NOTE watershed segmentation on the sensor channel doesn't work so great; for now we just run
    # the nucleus segmentation + dilate the nuclei
    # job_list = watershed_segmentation_workflow(config, seg_in_key, nuc_seg_in_key, job_list,
    #                                            erode_mask=20, dilate_seeds=3)
    job_list = nucleus_segmentation_workflow(config, nuc_seg_in_key, job_list,
                                             dilation_radius=config.dilation_radius,
                                             remove_nucleus_from_dilated=config.remove_nucleus_from_dilated,
                                             min_nucleus_size=config.min_nucleus_size,
                                             erosion_radius=config.erosion_radius)

    # image_outlier_criteria = {'max_number_cells': 1000,
    #                           'min_number_cells': 25}

    # add the qc and feature extraction tasks
    job_list.extend([
        (InstanceFeatureExtraction, {
            'build': {
                'channel_keys': ['sensor'],
                'cell_seg_key': config.nuc_key_eroded,
                'topk': []
            },
            'run': {
                'gpu_id': config.gpu,
                'on_cluster': config.on_cluster
            }
        }),
        (InstanceFeatureExtraction, {
            'build': {
                'channel_keys': ['infection marker',
                                 'infection marker2'],
                'cell_seg_key': config.nuc_key_dilated,
                'topk': []
            },
            'run': {
                'gpu_id': config.gpu,
                'on_cluster': config.on_cluster
            }
        }),
        # # need to add qc task(s) if we decide to make this assay more quantitative
        # (ImageLevelQC, {
        #     'build': {
        #         'cell_seg_key': config.nuc_key_eroded,
        #         'serum_key': 'sensor',
        #         'marker_key': 'infection marker',
        #         'outlier_criteria': image_outlier_criteria
        #     }
        # }),
        (DrugScreenAnalysisCellTable, {
            'build': {
                'nucleus_seg_key': config.nuc_key,
                'backgrounds': backgrounds
            },
            'run': {
                'n_jobs': config.n_cpus
            }
        })
    ])

    return job_list


def run_drug_screen_analysis(config):
    """
    """
    name = 'DrugScreenAnalysisWorkflow'

    # to allow running on the cpu
    if config.gpu is not None and config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    work_dir = os.path.join(config.folder, 'batchlib')
    os.makedirs(work_dir, exist_ok=True)

    nuc_seg_in_key = 'nuclei'
    job_list = core_ds_workflow_tasks(config, nuc_seg_in_key)

    t0 = time.time()
    run_workflow(name,
                 config.folder,
                 job_list,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs,
                 skip_processed=bool(config.skip_processed))
    print("Run drug-screen analysis workflow in", time.time() - t0, "s")

    if config.export_table:
        export_cell_table(config.folder)


def drug_screen_parser(config_folder, default_config_name):
    """
    """

    doc = """Run workflow for analysis of covid-if drug screen.
    Based on UNet boundary and foreground predictions,
    stardist nucleus prediction and watershed segmentation.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    mischelp = """Path to the folder batchlib/misc,
    that contains all necessary additional data to run the workflow"""

    default_config = os.path.join(config_folder, default_config_name)
    parser = configargparse.ArgumentParser(description=doc,
                                           default_config_files=[default_config],
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)

    # mandatory
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--input_folder', required=True, type=str, help='folder with input files as tifs')
    parser.add('--gpu', required=True, type=int, help='id of gpu for this job')
    parser.add('--n_cpus', required=True, type=int, help='number of cpus')
    parser.add('--folder', required=True, type=str, default="", help=fhelp)
    parser.add('--misc_folder', required=True, type=str, help=mischelp)
    parser.add('--is_wf', required=True, type=int, help="Is this widefield or confocal?")
    parser.add('--barrel_corrector_folder', type=str, default='auto',
               help='optinally specify the folder containing the files for barrel correction.')

    # folder options
    # this parameter is not necessary here any more, but for now we need it to be
    # compatible with the pixel-wise workflow
    parser.add("--output_root_name", default='data-processed')
    parser.add("--use_unique_output_folder", default=False)

    # keys for intermediate data
    parser.add("--bd_key", default='boundaries', type=str)
    parser.add("--mask_key", default='mask', type=str)
    parser.add("--nuc_key", default='nucleus_segmentation', type=str)
    parser.add("--nuc_key_dilated", default='nucleus_segmentation_dilated', type=str)
    parser.add("--nuc_key_eroded", default='nucleus_segmentation_eroded', type=str)

    #
    # more options
    #

    # segmentation options
    parser.add("--dilation_radius", default=5, type=int)
    parser.add("--erosion_radius", default=2, type=int)
    parser.add("--remove_nucleus_from_dilated", default=True)
    parser.add("--min_nucleus_size", default=50, type=int)

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)
    parser.add("--skip_processed", default=0, type=int)

    # additional image output
    # parser.add("--write_summary_images", default=True)
    # parser.add("--write_background_images", default=True)

    default_scale_factors = [1, 2, 4, 8, 16]
    parser.add("--scale_factors", default=default_scale_factors)

    # do we run on cluster?
    parser.add("--on_cluster", type=int, default=0)
    # do we export the tables for easier downstream analysis?
    parser.add("--export_table", type=int, default=1)

    return parser
