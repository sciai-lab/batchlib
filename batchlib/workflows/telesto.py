import os
import time

import configargparse
import pandas as pd

from batchlib import run_workflow
from batchlib.analysis.feature_extraction import InstanceFeatureExtraction
from batchlib.analysis.telesto_analysis import ImageLevelFeatures, MergeImageLevelFeatures
from batchlib.segmentation.segmentation_workflows import watershed_segmentation_workflow
from batchlib.preprocessing.preprocess_telesto import PreprocessTelesto
from batchlib.util import open_file, read_table
from batchlib.util.logger import get_logger

logger = get_logger('Workflow.CellAnalysis')


def get_initial_table(config):
    folder = config.input_folder
    name = os.path.split(folder)[1]
    initial_table_path = os.path.join(folder, f'{name}.tsv')
    return initial_table_path if os.path.exists(initial_table_path) else None


def core_telesto_workflow_tasks(config, seg_in_key, nuc_seg_in_key):
    job_list = [
        (PreprocessTelesto.from_folder, {
            'build': {
                'input_folder': config.input_folder,
                'barrel_corrector_path': None,
                'scale_factors': config.scale_factors},
            'run': {
                'n_jobs': config.n_cpus}})
    ]

    # TODO find the minimal nucleus size
    min_nuc_size = None
    job_list = watershed_segmentation_workflow(config, seg_in_key, nuc_seg_in_key, job_list,
                                               erode_mask=20, dilate_seeds=3, min_nucleus_size=min_nuc_size)

    serum_keys = ('serum_IgG', 'serum_IgA')
    # add the feature extraction tasks
    job_list.append(
        (InstanceFeatureExtraction, {
            'build': {
                'channel_keys': serum_keys,
                'nuc_seg_key_to_ignore': None,  # TODO for proper sum based features we would need to ignore here
                'cell_seg_key': config.seg_key
            },
            'run': {
                'gpu_id': config.gpu,
                'on_cluster': config.on_cluster
            }
        })
    )

    # the initial table with metadata information.
    # only available for the training dataset
    initial_table_path = get_initial_table(config)

    # compute the image level features for IgA and IgG
    for serum_key in serum_keys:
        job_list.append(
            (ImageLevelFeatures, {
                'build': {
                    'cell_seg_key': config.seg_key,
                    'serum_key': serum_key,
                    'initial_table_path': initial_table_path
                }
            })
        )

    # merge the features
    job_list.append(
        (MergeImageLevelFeatures, {
            'build': {
                'input_table_names': serum_keys,
                'initial_table_path': initial_table_path
            }
        })
    )

    return job_list


def export_image_table(folder, plate_name=None):
    if plate_name is None:
        plate_name = os.path.split(folder.rstrip('/'))[1]
    in_table_path = os.path.join(folder, f'{plate_name}_table.hdf5')
    out_table_path = os.path.join(folder, f'{plate_name}_table.xlsx')

    with open_file(in_table_path, 'r') as f:
        cols, tab = read_table(f, 'images/default')

    df = pd.DataFrame(tab, columns=cols)
    df.to_excel(out_table_path, index=False)


def run_telesto_analysis(config):
    """
    """
    name = 'TelestoAnalysisWorkflow'

    # to allow running on the cpu
    if config.gpu is not None and config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('telesto', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    work_dir = os.path.join(config.folder, 'batchlib')
    os.makedirs(work_dir, exist_ok=True)

    seg_in_key = 'serum_IgG'
    nuc_seg_in_key = 'nuclei'
    job_list = core_telesto_workflow_tasks(config, seg_in_key, nuc_seg_in_key)

    t0 = time.time()
    run_workflow(name,
                 config.folder,
                 job_list,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs,
                 skip_processed=bool(config.skip_processed))
    print("Run telesto analysis workflow in", time.time() - t0, "s")

    # export excel table for the image level features
    if config.export_table:
        export_image_table(config.folder)


def telesto_parser(config_folder, default_config_name):
    """
    """

    doc = """Run workflow for analysis of teleto if data.
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
    # parser.add('--barrel_corrector_folder', type=str, default='auto',
    #            help='optinally specify the folder containing the files for barrel correction.')

    # folder options
    # this parameter is not necessary here any more, but for now we need it to be
    # compatible with the pixel-wise workflow
    parser.add("--output_root_name", default='telesto/data-processed')
    parser.add("--use_unique_output_folder", default=False)

    # keys for intermediate data
    parser.add("--bd_key", default='boundaries', type=str)
    parser.add("--mask_key", default='mask', type=str)
    parser.add("--nuc_key", default='nucleus_segmentation', type=str)
    parser.add("--seg_key", default='cell_segmentation', type=str)

    #
    # more options
    #

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)
    parser.add("--skip_processed", default=0, type=int)

    default_scale_factors = [1, 2, 4, 8, 16]
    parser.add("--scale_factors", default=default_scale_factors)

    # do we run on cluster?
    parser.add("--on_cluster", type=int, default=0)
    # do we export the tables for easier downstream analysis?
    parser.add("--export_table", type=int, default=1)

    return parser
