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
from batchlib.segmentation.voronoi_ring_segmentation import ErodeSegmentation
from batchlib.reporting import SlackSummaryWriter, export_tables_for_plate
from batchlib.util import get_logger
from batchlib.util.plate_visualizations import all_plots

logger = get_logger('Workflow.CellAnalysis')

DEFAULT_PLOT_NAMES = ['serum_ratio_of_q0.5_of_means',
                      'serum_ratio_of_q0.5_of_sums',
                      'serum_robust_z_score_sums',
                      'serum_robust_z_score_means']


def get_analysis_parameter(config, background_parameters):
    # collect all relevant analysis paramter, so that we can
    # write them to a table and keep track of this
    params = {'marker_denoise_radius': config.marker_denoise_radius,
              'ignore_nuclei': config.ignore_nuclei,
              'infected_detection_threshold': config.infected_threshold,
              'scale_infected_detection_with_mad': config.infected_scale_with_mad}

    params.update({'qc_cells_' + k: v for k, v in DEFAULT_CELL_OUTLIER_CRITERIA.items()})
    params.update({'qc_images_' + k: v for k, v in DEFAULT_IMAGE_OUTLIER_CRITERIA.items()})
    params.update({'qc_wells_' + k: v for k, v in DEFAULT_WELL_OUTLIER_CRITERIA.items()})
    params.update({'background_' + k: v for k, v in background_parameters.items()})
    return params


def get_input_keys(config, serum_in_keys):

    nuc_in_key = 'nuclei'
    marker_in_key = 'marker'

    # keys for the segmentation tasks
    # compute segmentation on IgG if available
    try:
        serum_seg_in_key = next(iter(filter(lambda key: key.endswith('IgG'), serum_in_keys)))
    except StopIteration:
        serum_seg_in_key = serum_in_keys[0]
    nuc_seg_in_key = nuc_in_key

    # keys for the analysis tasks
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
    background_dict = config.background_dict
    if background_dict is None:
        logger.info(f"Compute background from data and use the per plate background")
        return {key: f'plate/backgrounds' for key in keys}

    if isinstance(background_dict, str):
        background_dict = json.loads(background_dict.replace('\'', '\"'))
    assert isinstance(background_dict, dict)
    validate_bg_dict(background_dict)

    if len(set(keys) - set(background_dict.keys())) > 0:
        bg_keys = list(background_dict.keys())
        raise ValueError(f"Did not find values for all chennales {keys} in {bg_keys}")
    return background_dict


def core_workflow_tasks(config, name, feature_identifier):

    # to allow running on the cpu
    if config.gpu is not None and config.gpu < 0:
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
    if not os.path.exists(barrel_corrector_path):
        raise ValueError(f"Invalid barrel corrector path {barrel_corrector_path}")

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

    seg_key_for_infected_classification = config.seg_key + '_for_infected_classification'

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
                'radius': config.marker_denoise_radius}}))
        marker_ana_in_key = marker_ana_in_key + '_denoised'

    # add the tasks to extract the features from the cell instance segmentation,
    # do initial image level qc (necessary for the background extraction).
    # extract the background and find the infected cells
    job_list.extend(
        [(InstanceFeatureExtraction, {
          'build': {
              'channel_keys': (*serum_ana_in_keys, marker_ana_in_key),
              'nuc_seg_key_to_ignore': None if config.ignore_nuclei else config.nuc_key,
              'cell_seg_key': config.seg_key,
              'identifier': feature_identifier},
          'run': {'gpu_id': config.gpu}}),
         (ImageLevelQC, {
          'build': {
              'cell_seg_key': config.seg_key,
              'serum_key': serum_seg_in_key,
              'marker_key': marker_ana_in_key,
              'feature_identifier': feature_identifier,
              'outlier_predicate': outlier_predicate}}),
         # NOTE the bg extraction is independent of the features, but we still need to pass
         # the identifier so that the input validation passes
         (ExtractBackground, {
          'build': {
              'marker_key': marker_ana_in_key,  # is ignored
              'serum_key': serum_seg_in_key,    # is ignored
              'actual_channels_to_use': (*serum_ana_in_keys, marker_ana_in_key),  # is actually used
              'feature_identifier': feature_identifier,
              'cell_seg_key': config.seg_key}}),
         (ErodeSegmentation, {
             'build': {
                 'input_key': config.seg_key,
                 'output_key': seg_key_for_infected_classification,
                 'radius': config.erosion_radius,  # default: 2
             }
         }),
         # this one is just for the infected cell classification
         (InstanceFeatureExtraction, {
             'build': {
                 'channel_keys': (marker_ana_in_key,),
                 'nuc_seg_key_to_ignore': config.nuc_key,  # TODO add all the infected cell detection params to parser
                 'cell_seg_key': seg_key_for_infected_classification,
                 'identifier': None,
                 'quantiles': [0.93],
             },
             'run': {'gpu_id': config.gpu}}),
         # NOTE we need to pass the feature identifier, but the infected cells will only be
         # computed the first time
         # TODO: make sure that the quantile is there
         (FindInfectedCells, {
          'build': {
              'marker_key': marker_ana_in_key,
              'cell_seg_key': seg_key_for_infected_classification,
              'scale_with_mad': config.infected_scale_with_mad,  # default: True
              'infected_threshold': config.infected_threshold,  # default: 5.4
              'split_statistic': 'quantile0.93',
              'feature_identifier': None,
              'bg_correction_key': 'plate/backgrounds'}})]
    )

    # for the background substraction, we can either use a fixed value per channel,
    # or compute it from the data. In the first case, we pass the value
    # as 'serum_bg_key' / 'marker_bg_key'. In the second case, we pass a key
    # to the table holding these values.
    background_parameters = parse_background_parameters(config, marker_ana_in_key, serum_ana_in_keys)

    table_identifiers = serum_ana_in_keys if feature_identifier is None else [k + f'_{feature_identifier}'
                                                                              for k in serum_ana_in_keys]
    # NOTE currently the QC tasks will not be rerun if the feature identifier changes
    for serum_key, identifier in zip(serum_ana_in_keys, table_identifiers):
        job_list.append((CellLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'feature_identifier': feature_identifier,
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
                'feature_identifier': feature_identifier,
                'identifier': identifier}
        }))
        job_list.append((WellLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'feature_identifier': feature_identifier,
                'identifier': identifier}
        }))
        job_list.append((CellLevelAnalysis, {
            'build': {
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'cell_seg_key': config.seg_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'write_summary_images': config.write_summary_images,
                'scale_factors': config.scale_factors,
                'feature_identifier': feature_identifier,
                'identifier': identifier},
            'run': {'force_recompute': False}}))

    # get a dict with all relevant analysis parameters, so that we can write it as a table and log it
    analysis_parameter = get_analysis_parameter(config, background_parameters)
    logger.info(f"Analysis parameter: {analysis_parameter}")

    # find the identifier for the reference table (the IgG one if we have multiple tables)
    if len(table_identifiers) == 1:
        reference_table_name = table_identifiers[0]
    else:
        reference_table_name = [table_id for table_id in table_identifiers if 'IgG' in table_id]
        assert len(reference_table_name) == 1, f"{table_identifiers}"
        reference_table_name = reference_table_name[0]

    job_list.append((MergeAnalysisTables, {
        'build': {'input_table_names': table_identifiers,
                  'reference_table_name': reference_table_name,
                  'analysis_parameters': analysis_parameter,
                  'identifier': feature_identifier}
    }))

    return job_list, table_identifiers


def workflow_summaries(name, config, table_identifiers, t0, stat_names=DEFAULT_PLOT_NAMES):
    # run all plots on the output files
    plot_folder = os.path.join(config.folder, 'plots')

    for identifier in table_identifiers:
        table_path = CellLevelAnalysis.folder_to_table_path(config.folder)
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

    db_writer = DbResultWriter(
        username=config.db_username,
        password=config.db_password,
        host=config.db_host,
        port=config.db_port,
        db_name=config.db_name
    )
    db_writer(config.folder, config.folder)

    t0 = time.time() - t0
    summary_writer = SlackSummaryWriter(config.slack_token)
    summary_writer(config.folder, config.folder, runtime=t0)

    if config.export_tables:
        export_tables_for_plate(config.folder)
    logger.info(f"Run {name} in {t0}s")


def run_cell_analysis(config):
    """
    """
    name = 'CellAnalysisWorkflow'
    feature_identifier = config.feature_identifier
    if feature_identifier is not None:
        name += f'_{feature_identifier}'

    job_list, table_identifiers = core_workflow_tasks(config, name, feature_identifier)

    t0 = time.time()
    run_workflow(name,
                 config.folder,
                 job_list,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs)

    # only run the workflow summaries if we don't have the feature identifier
    if feature_identifier is None:
        workflow_summaries(name, config, table_identifiers, t0)

    return table_identifiers


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

    # we can set an additional feature identifier, to make several runs
    # of the cell analysis workflow unique
    # if this is done, the full workflow will not be run, only up until MergeTables
    parser.add("--feature_identifier", type=str, default=None)

    #
    # analysis parameter:
    # these parameters change how the analysis results are computed!
    #

    # marker denoising, segmentation erosion (only for cell classifictaion) and ignore nuclei
    parser.add("--marker_denoise_radius", default=0, type=int)
    parser.add("--erosion_radius", default=2, type=int)
    parser.add("--ignore_nuclei", default=True)

    # parameter for the infected cell detection
    parser.add("--infected_scale_with_mad", default=True)
    parser.add("--infected_threshold", type=float, default=4.2)

    # optional background subtraction values for the individual channels
    # if None, all backgrounds will be computed from the data and the plate background will be used
    parser.add("--background_dict", default=None)

    #
    # more options
    #

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)
    parser.add("--write_summary_images", default=True)

    # MongoDB client config
    parser.add("--db_username", type=str, default='covid19')
    parser.add("--db_password", type=str, default='')
    parser.add("--db_host", type=str, default='localhost')
    parser.add("--db_port", type=int, default=27017)
    parser.add("--db_name", type=str, default='covid')

    # slack client
    parser.add("--slack_token", type=str, default=None)

    default_scale_factors = [1, 2, 4, 8, 16]
    parser.add("--scale_factors", default=default_scale_factors)

    # do we run on cluster?
    parser.add("--on_cluster", type=int, default=0)
    # do we export the tables for easier downstream analysis?
    parser.add("--export_tables", type=int, default=0)

    return parser
