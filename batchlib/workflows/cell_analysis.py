import os
import json
import time
from numbers import Number

import configargparse

from batchlib import run_workflow
from batchlib.analysis.cell_level_analysis import (CellLevelAnalysis,
                                                   DenoiseByGrayscaleOpening,
                                                   InstanceFeatureExtraction,
                                                   FindInfectedCells,
                                                   ExtractBackground)
from batchlib.analysis.cell_analysis_qc import (CellLevelQC, ImageLevelQC, WellLevelQC,
                                                DEFAULT_CELL_OUTLIER_CRITERIA,
                                                DEFAULT_IMAGE_OUTLIER_CRITERIA,
                                                DEFAULT_WELL_OUTLIER_CRITERIA)
from batchlib.analysis.merge_tables import MergeAnalysisTables
from batchlib.mongo.result_writer import DbResultWriter
from batchlib.outliers.outlier import get_outlier_predicate
from batchlib.preprocessing import get_barrel_corrector, get_serum_keys, Preprocess
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.reporting import SlackSummaryWriter
from batchlib.util import get_logger
from batchlib.util.plate_visualizations import all_plots

logger = get_logger('Workflow.CellAnalysis')


def get_analysis_parameter(config, use_fixed_background, background_parameters):
    # collect all relevant analysis paramter, so that we can
    # write them to a table and keep track of this
    params = {'marker_denoise_radius': config.marker_denoise_radius,
              'dont_ignore_nuclei': config.dont_ignore_nuclei,
              'infected_detection_threshold': config.infected_threshold,
              'scale_infected_detection_with_mad': config.infected_scale_with_mad}
    params.update({'qc_cells_' + k: v for k, v in DEFAULT_CELL_OUTLIER_CRITERIA.items()})
    params.update({'qc_images_' + k: v for k, v in DEFAULT_IMAGE_OUTLIER_CRITERIA.items()})
    params.update({'qc_wells_' + k: v for k, v in DEFAULT_WELL_OUTLIER_CRITERIA.items()})

    params['fixed_background'] = use_fixed_background
    if use_fixed_background:
        params.update({'background_' + name: value for name, value in background_parameters.items()})
    else:
        params.update({'background_type': config.background_type})

    return params


def get_input_keys(config, serum_in_keys):

    nuc_in_key = 'nuclei'
    marker_in_key = 'marker'

    # compute segmentation on IgG if available
    try:
        serum_seg_in_key = next(iter(filter(lambda key: key.endswith('IgG'), serum_in_keys)))
    except StopIteration:
        serum_seg_in_key = serum_in_keys[0]

    if config.segmentation_on_corrected:
        nuc_seg_in_key = nuc_in_key + '_corrected'
        serum_seg_in_key = serum_seg_in_key + '_corrected'
    else:
        nuc_seg_in_key = nuc_in_key

    if config.analysis_on_corrected:
        serum_ana_in_keys = [serum_in_key + '_corrected' for serum_in_key in serum_in_keys]
        marker_ana_in_key = marker_in_key + '_corrected'
    else:
        serum_ana_in_keys = serum_in_keys
        marker_ana_in_key = marker_in_key

    return nuc_seg_in_key, serum_seg_in_key, marker_ana_in_key, serum_ana_in_keys


def validate_bg_dict(bg_dict):
    excepted_vals = ('images/backgrounds',
                     'wells/backgrounds',
                     'plate/backgrounds')
    for key, val in bg_dict.items():
        if not isinstance(val, Number) and val not in excepted_vals:
            raise ValueError(f"Invalid background value {val} for {key}")


def parse_background_parameters(config, marker_ana_in_key, serum_ana_in_keys):
    keys = serum_ana_in_keys + [marker_ana_in_key]
    fixed_background_dict = config.fixed_background
    if fixed_background_dict is None:
        background_type = config.background_type
        if background_type not in ('images', 'wells', 'plate'):
            raise ValueError(f"Expected background type to be one of (images, wells, plates), got {background_type}")
        logger.info(f"Compute background from data with type {background_type}")
        return {key: f'{background_type}/backgrounds' for key in keys}, False

    fixed_background_dict = json.loads(fixed_background_dict.replace('\'', '\"'))
    assert isinstance(fixed_background_dict, dict)

    # TODO I don't want to enforce having the _corrected everywhere, so we do this weird little dance for now
    # in the long term, I would like to eliminate running the option of using the non-corrected completely,
    # then we can get rid of this
    parsed_fixed_background_dict = {}
    for key in keys:
        value = fixed_background_dict.get(key, None)
        if value is None:
            alt_key = key.replace('_corrected', '')
            value = fixed_background_dict.get(alt_key, None)
        if value is None:
            raise ValueError(f"Could not find fixed background value for channel {key}")
        parsed_fixed_background_dict[key] = value

    validate_bg_dict(parsed_fixed_background_dict)
    logger.info(f"Use fixed backgrounds: {parsed_fixed_background_dict}")
    return parsed_fixed_background_dict, True


def run_cell_analysis(config):
    """
    """
    assert (config.dont_ignore_nuclei is False), "We need to run computation WITH nucleus exclusion"

    name = 'CellAnalysisWorkflow'

    # to allow running on the cpu
    if config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    misc_folder = config.misc_folder

    model_root = os.path.join(misc_folder, 'models/stardist')
    model_name = '2D_dsb2018'

    barrel_corrector_path = get_barrel_corrector(os.path.join(misc_folder, 'barrel_correctors'),
                                                 config.input_folder)

    torch_model_path = os.path.join(misc_folder, 'models/torch/fg_and_boundaries_V1.torch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    # get keys and identifier
    serum_in_keys = get_serum_keys(config.input_folder)
    (nuc_seg_in_key, serum_seg_in_key,
     marker_ana_in_key, serum_ana_in_keys) = get_input_keys(config, serum_in_keys)

    outlier_predicate = get_outlier_predicate(config)

    job_list = [
        (Preprocess.from_folder, {
            'build': {
                'input_folder': config.input_folder,
                'barrel_corrector_path': barrel_corrector_path,
                'scale_factors': config.scale_factors},
            'run': {
                'n_jobs': config.n_cpus}}),
        (TorchPrediction, {
            'build': {
                'input_key': serum_seg_in_key,
                'output_key': [config.mask_key, config.bd_key],
                'model_path': torch_model_path,
                'model_class': torch_model_class,
                'model_kwargs': torch_model_kwargs,
                'scale_factors': config.scale_factors},
            'run': {
                'gpu_id': config.gpu,
                'batch_size': config.batch_size,
                'threshold_channels': {0: 0.5},
                'on_cluster': config.on_cluster}}),
        (StardistPrediction, {
            'build': {
                'model_root': model_root,
                'model_name': model_name,
                'input_key': nuc_seg_in_key,
                'output_key': config.nuc_key,
                'scale_factors': config.scale_factors},
            'run': {
                'gpu_id': config.gpu,
                'n_jobs': config.n_cpus,
                'on_cluster': config.on_cluster}}),
        (SeededWatershed, {
            'build': {
                'pmap_key': config.bd_key,
                'seed_key': config.nuc_key,
                'output_key': config.seg_key,
                'mask_key': config.mask_key,
                'scale_factors': config.scale_factors},
            'run': {
                'erode_mask': 20,
                'dilate_seeds': 3,
                'n_jobs': config.n_cpus}})
    ]
    if config.marker_denoise_radius > 0:
        job_list.append((DenoiseByGrayscaleOpening, {
            'build': {
                'key_to_denoise': marker_ana_in_key,
                'radius': config.marker_denoise_radius},
            'run': {}}))
        marker_ana_in_key = marker_ana_in_key + '_denoised'

    job_list.append((InstanceFeatureExtraction, {
        'build': {
            'channel_keys': (*serum_ana_in_keys, marker_ana_in_key),
            'nuc_seg_key_to_ignore': config.nuc_key if not config.dont_ignore_nuclei else None,
            'cell_seg_key': config.seg_key},
        'run': {'gpu_id': config.gpu}}))

    # This is just for ExtractBackground below
    job_list.append((ImageLevelQC, {
        'build': {
            'cell_seg_key': config.seg_key,
            'serum_key': serum_seg_in_key,
            'marker_key': marker_ana_in_key,
            'outlier_predicate': outlier_predicate,
            'identifier': None}
    }))

    job_list.append((ExtractBackground, {
        'build': {
            'marker_key': marker_ana_in_key,  # is ignored
            'serum_key': serum_seg_in_key,    # is ignored
            'cell_seg_key': config.seg_key,
            'actual_channels_to_use': (*serum_ana_in_keys, marker_ana_in_key),  # is actually used
        }
    }))
    # # Also compute features with nuclei if they should be used later
    # job_list.append((InstanceFeatureExtraction, {
    #     'build': {
    #         'channel_keys': (*serum_ana_in_keys, marker_ana_in_key),
    #         'nuc_seg_key_to_ignore': None,
    #         'identifier': 'with_nuclei',
    #         'cell_seg_key': config.seg_key},
    #     'run': {'gpu_id': config.gpu}}))

    job_list.append((FindInfectedCells, {
        'build': {
            'marker_key': marker_ana_in_key,
            'cell_seg_key': config.seg_key,
            'bg_correction_key': 'plate/backgrounds',
            'scale_with_mad': config.infected_scale_with_mad,  # default: True
            'infected_threshold': config.infected_threshold  # default: 6.2
        }}))

    # for the background substraction, we can either use a fixed value,
    # or compute it from the data. In the first case, we pass the value
    # as 'serum_bg_key' / 'marker_bg_key'. In the second case, we pass a key
    # to the table holding these values.
    background_parameters, is_fixed = parse_background_parameters(config,
                                                                  marker_ana_in_key,
                                                                  serum_ana_in_keys)

    table_identifiers = serum_ana_in_keys
    for serum_key, identifier in zip(serum_ana_in_keys, table_identifiers):
        job_list.append((CellLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'identifier': identifier}
        }))
        job_list.append((ImageLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'outlier_predicate': outlier_predicate,
                'identifier': identifier}
        }))
        job_list.append((WellLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'identifier': identifier}
        }))
        job_list.append((CellLevelAnalysis, {
            'build': {
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'cell_seg_key': config.seg_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'write_summary_images': False,
                'scale_factors': config.scale_factors,
                'identifier': identifier},
            'run': {'force_recompute': False}}))

    # get a dict with all relevant analysis parameters, so that we can write it as a table and log it
    # TODO we also need to include this in the database
    analysis_parameter = get_analysis_parameter(config, is_fixed, background_parameters)
    logger.info(f"Analysis parameter: {analysis_parameter}")

    # find the identifier for the reference table (the IgG one if we have multiple tables)
    if len(table_identifiers) == 1:
        reference_table_name = table_identifiers[0]
    else:
        reference_table_name = [table_id for table_id in table_identifiers if 'IgG' in table_id]
        assert len(reference_table_name) == 1, f"{table_identifiers}"
        reference_table_name = reference_table_name[0]

    # TODO
    # - we need to filter out the mean-with-nuclei and sum-without-nuclei results (not implemented yet)
    job_list.append((MergeAnalysisTables, {
        'build': {'input_table_names': table_identifiers,
                  'reference_table_name': reference_table_name,
                  'analysis_parameters': analysis_parameter}
    }))

    # make sure that db job is executed when all result tables hdf5 are ready (outside of the loop)
    job_list.append((DbResultWriter, {
        'build': {
            "username": config.db_username,
            "password": config.db_password,
            "host": config.db_host,
            "port": config.db_port,
            "db_name": config.db_name
        }}))

    t0 = time.time()

    run_workflow(name,
                 config.folder,
                 job_list,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)

    # run all plots on the output files
    plot_folder = os.path.join(config.folder, 'plots')
    stat_names = ['serum_ratio_of_q0.5_of_means',
                  'serum_ratio_of_q0.5_of_sums',
                  'serum_robust_z_score_sums',
                  'serum_robust_z_score_means']
    for identifier in table_identifiers:
        table_path = CellLevelAnalysis.folder_to_table_path(config.folder, identifier)
        all_plots(table_path, plot_folder,
                  table_key=f'images/{identifier}',
                  identifier=identifier + '_per-image',
                  stat_names=stat_names,
                  channel_name=identifier,
                  wedge_width=0.3)
        all_plots(table_path, plot_folder,
                  table_key=f'wells/{identifier}',
                  identifier=identifier + '_per-well',
                  stat_names=stat_names,
                  channel_name=identifier,
                  wedge_width=0)

    t0 = time.time() - t0

    summary_writer = SlackSummaryWriter(config.slack_token)
    summary_writer(config.folder, config.folder, runtime=t0)

    logger.info(f"Run {name} in {t0}s")
    return name, t0


def cell_analysis_parser(config_folder, default_config_name):
    """
    """

    doc = """Run cell based analysis workflow.
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

    # folder options
    # this parameter is not necessary here any more, but for now we need it to be
    # compatible with the pixel-wise workflow
    parser.add("--root", default='/home/covid19/antibodies-nuclei')
    parser.add("--output_root_name", default='data-processed')
    parser.add("--use_unique_output_folder", default=False)

    # keys for intermediate data
    parser.add("--bd_key", default='boundaries', type=str)
    parser.add("--mask_key", default='mask', type=str)
    parser.add("--nuc_key", default='nucleus_segmentation', type=str)
    parser.add("--seg_key", default='cell_segmentation', type=str)

    #
    # analysis parameter:
    # these parameters change how the analysis results are computed!
    #

    # marker denoising and ignore nuclei
    parser.add("--marker_denoise_radius", default=0, type=int)
    parser.add("--dont_ignore_nuclei", action='store_true')

    # parameter for the infected cell detection
    parser.add("--infected_scale_with_mad", default=True)
    parser.add("--infected_threshold", type=float, default=6.2)

    # optional fixed background value(s).
    # if None, will be computed from the data
    # otherwise, needs to be a dict with background values for the input channels
    parser.add("--fixed_background", default=None)
    # do we use image, well or plate level background? (default is plate)
    parser.add("--background_type", default='plate', type=str)

    #
    # more options
    #

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)

    # MongoDB client config
    parser.add("--db_username", type=str, default='covid19')
    parser.add("--db_password", type=str, default='')
    parser.add("--db_host", type=str, default='localhost')
    parser.add("--db_port", type=int, default=27017)
    parser.add("--db_name", type=str, default='covid')

    # slack client
    parser.add("--slack_token", type=str, default=None)

    # default_scale_factors = None
    default_scale_factors = [1, 2, 4, 8, 16]
    parser.add("--scale_factors", default=default_scale_factors)

    # do we run on cluster?
    parser.add("--on_cluster", type=int, default=0)

    return parser
