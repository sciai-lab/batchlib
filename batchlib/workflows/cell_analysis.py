import os
import json
import time
from numbers import Number

import configargparse

from batchlib import run_workflow

from batchlib.analysis.cell_level_analysis import (CellLevelAnalysis,
                                                   DenoiseByGrayscaleOpening,
                                                   DenoiseByWhiteTophat,
                                                   FindInfectedCells)
from batchlib.analysis.background_extraction import (ExtractBackground,
                                                     BackgroundFromWells)
from batchlib.analysis.cell_analysis_qc import (CellLevelQC, ImageLevelQC, WellLevelQC,
                                                DEFAULT_CELL_OUTLIER_CRITERIA,
                                                DEFAULT_IMAGE_OUTLIER_CRITERIA,
                                                DEFAULT_WELL_OUTLIER_CRITERIA)
from batchlib.analysis.feature_extraction import (InstanceFeatureExtraction,
                                                  SegmentationProperties)
from batchlib.analysis.merge_tables import MergeAnalysisTables

from batchlib.mongo.result_writer import DbResultWriter
from batchlib.outliers.outlier import get_outlier_predicate
from batchlib.preprocessing import get_barrel_corrector, get_serum_keys, Preprocess
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.segmentation.voronoi_ring_segmentation import ErodeSegmentation  # , VoronoiRingSegmentation
from batchlib.reporting import (SlackSummaryWriter,
                                export_tables_for_plate,
                                WriteBackgroundSubtractedImages)
from batchlib.util import get_logger, open_file, read_table, has_table
from batchlib.util.plate_visualizations import all_plots

logger = get_logger('Workflow.CellAnalysis')

DEFAULT_PLOT_NAMES = ['ratio_of_q0.5_of_means',
                      'ratio_of_q0.5_of_sums',
                      'robust_z_score_sums',
                      'robust_z_score_means']

# these are the default min serum intensities that are used for QC, if we DO NOT have
# empty wells.
# the intensity thresholds are derived from 3 * mad background, see
# https://github.com/hci-unihd/antibodies-analysis-issues/issues/84#issuecomment-632658726
DEFAULT_MIN_SERUM_INTENSITIES = {'serum_IgG': 301.23, 'serum_IgA': 392.76, 'serum_IgM': None}


def get_analysis_parameter(config, background_parameters):
    # collect all relevant analysis paramter, so that we can
    # write them to a table and keep track of this
    params = {'marker_denoise_radius': config.marker_denoise_radius,
              'ignore_nuclei': config.ignore_nuclei,
              'infected_detection_threshold': config.infected_threshold,
              'infected_detection_erosion_radius': config.infected_erosion_radius,
              'infected_detection_quantile': config.infected_quantile,
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
                     'plate/backgrounds',
                     'plate/backgrounds_from_min_well')
    for key, val in bg_dict.items():
        if not isinstance(val, Number) and val not in excepted_vals:
            raise ValueError(f"Invalid background value {val} for {key}")


def default_bg_parameters(config, keys):
    name = os.path.split(config.input_folder)[1]

    bg_info_dict = os.path.join(config.misc_folder, 'plates_to_background_well_new.json')
    with open(bg_info_dict) as f:
        bg_info_dict = json.load(f)

    # all new plates have H01 and G01 as empty wells for bg estimation
    bg_info = bg_info_dict.get(name, ['H01', 'G01'])

    # the bg-info can have 2 different structures:
    # list: -> list of wells for background estimation
    # dict: -> dict of fixed background values for the channels
    if isinstance(bg_info, list):
        bg_wells = bg_info
        # NOTE: marker background needs to be estimated from the whole plate
        bg_dict = {k: 'plate/backgrounds_min_well' if k.startswith('serum') else 'plate/backgrounds'
                   for k in keys}
    else:
        bg_wells = []
        bg_dict = {k: bg_info.get(k, 'plate/backgrounds') for k in keys}

    return bg_dict, bg_wells


def parse_background_parameters(config, marker_ana_in_key, serum_ana_in_keys):
    keys = serum_ana_in_keys + [marker_ana_in_key]
    background_dict = config.background_dict
    if background_dict is None:
        logger.info("Background parameter were not specified, using default values for this plate")
        return default_bg_parameters(config, keys)

    if isinstance(background_dict, str):
        background_dict = json.loads(background_dict.replace('\'', '\"'))
    assert isinstance(background_dict, dict)
    validate_bg_dict(background_dict)

    if len(set(keys) - set(background_dict.keys())) > 0:
        bg_keys = list(background_dict.keys())
        raise ValueError(f"Did not find values for all channels {keys} in {bg_keys}")
    return background_dict, []


def add_infected_detection_jobs(job_list, config, marker_key, feature_identifier):
    erosion_radius = config.infected_erosion_radius
    tophat_radius = config.infected_tophat_radius
    quantile = config.infected_quantile

    if erosion_radius > 0:
        seg_key_for_infected_classification = config.seg_key + '_for_infected_classification'
        job_list.append((ErodeSegmentation, {
            'build': {
                 'input_key': config.seg_key,
                 'output_key': seg_key_for_infected_classification,
                 'radius': erosion_radius},
            'run': {'n_jobs': config.n_cpus}
        }))
    else:
        seg_key_for_infected_classification = config.seg_key

    if erosion_radius > 0 \
            or tophat_radius > 0 \
            or config.ignore_nuclei_in_infected_classification != config.ignore_nuclei:
        job_list.append((InstanceFeatureExtraction, {
            'build': {
                'channel_keys': (marker_key,),
                'nuc_seg_key_to_ignore': config.nuc_key if config.ignore_nuclei_in_infected_classification else None,
                'cell_seg_key': seg_key_for_infected_classification,
                'identifier': None,
                'quantiles': [quantile]},
            'run': {'gpu_id': config.gpu, 'on_cluster': config.on_cluster}
        }))
        this_feature_identifier = None
    else:
        this_feature_identifier = feature_identifier

    link_out_table = config.seg_key

    job_list.append(
        (FindInfectedCells, {
         'build': {
             'marker_key': marker_key,
             'cell_seg_key': seg_key_for_infected_classification,
             'scale_with_mad': config.infected_scale_with_mad,  # default: True
             'infected_threshold': config.infected_threshold,  # default: 5.4
             'split_statistic': 'quantile' + str(quantile),
             'feature_identifier': this_feature_identifier,
             'bg_correction_key': 'plate/backgrounds',
             'link_out_table': link_out_table}})
    )
    return job_list


def add_background_estimation(job_list, seg_key, channel_keys, identifier=None):
    job_list.append((
         ExtractBackground, {
            'build': {'channel_keys': channel_keys,
                      'identifier': identifier,
                      'cell_seg_key': seg_key},
            'run': {'force_recompute': None}}
    ))
    return job_list


def add_background_estimation_from_min_well(job_list, config, wells, channel_keys):
    # add the background estimation jobs
    job_list.append((
        BackgroundFromWells, {
            'build': {'well_list': wells,
                      'output_table': 'plate/backgrounds_min_well',
                      'seg_key': config.seg_key,
                      'channel_names': channel_keys},
            "run": {"force_recompute": None}
            }
    ))
    return job_list


def get_barrel_corrector_folder(config):
    barrel_corrector_root = os.path.join(config.misc_folder, 'barrel_correctors')
    with open(os.path.join(barrel_corrector_root, 'plates_with_old_setup.json')) as f:
        plates_with_old_setup = json.load(f)

    plate_name = os.path.split(config.input_folder)[1]
    if plate_name in plates_with_old_setup:
        subfolder = 'old_microscope'
    else:
        subfolder = 'new_microscope'

    barrel_corrector_folder = os.path.join(barrel_corrector_root, subfolder)
    assert os.path.exists(barrel_corrector_folder)

    return barrel_corrector_folder


def core_workflow_tasks(config, name, feature_identifier):

    # to allow running on the cpu
    if config.gpu is not None and config.gpu < 0:
        config.gpu = None

    config.input_folder = os.path.abspath(config.input_folder)
    if config.folder == "":
        config.folder = config.input_folder.replace('covid-data-vibor', config.output_root_name)
        if config.use_unique_output_folder:
            config.folder += '_' + name

    model_root = os.path.join(config.misc_folder, 'models/stardist')
    model_name = '2D_dsb2018'

    if config.barrel_corrector_folder == 'auto':
        barrel_corrector_folder = get_barrel_corrector_folder(config)
    else:
        barrel_corrector_folder = config.barrel_corrector_folder
    barrel_corrector_path = get_barrel_corrector(barrel_corrector_folder, config.input_folder)
    if not os.path.exists(barrel_corrector_path):
        raise ValueError(f"Invalid barrel corrector path {barrel_corrector_path}")

    torch_model_path = os.path.join(config.misc_folder, 'models/torch/fg_and_boundaries_V2.torch')
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
                'n_jobs': config.n_cpus}}),
        (SegmentationProperties, {
            'build': {'seg_key': config.seg_key},
            'run': {'n_jobs': config.n_cpus}})
    ]

    # check whether we apply denoising to the marker before analysis
    if config.marker_denoise_radius > 0:
        job_list.append((DenoiseByGrayscaleOpening, {
            'build': {
                'key_to_denoise': marker_ana_in_key,
                'radius': config.marker_denoise_radius},
            'run': {'n_jobs': config.n_cpus}
        }))
        marker_ana_in_key = marker_ana_in_key + '_denoised'
    if config.infected_tophat_radius > 0:
        job_list.append((DenoiseByWhiteTophat, {
            'build': {
                'key_to_denoise': marker_ana_in_key,
                'radius': config.infected_tophat_radius,
                'output_key': marker_ana_in_key + '_tophat'},
            'run': {'n_jobs': config.n_cpus}
        }))
        marker_ana_in_key += '_tophat'

    # add the tasks to extract the features from the cell instance segmentation,
    # do initial image level qc (necessary for the background extraction) and extract the quantiles
    # + do the image QC necessary for the bg estimation jobs
    job_list.extend([
        (InstanceFeatureExtraction, {
            'build': {'channel_keys': (*serum_ana_in_keys, marker_ana_in_key),
                      'nuc_seg_key_to_ignore': None if config.ignore_nuclei else config.nuc_key,
                      'cell_seg_key': config.seg_key,
                      'quantiles': [config.infected_quantile],
                      'identifier': feature_identifier},
            'run': {'gpu_id': config.gpu, 'on_cluster': config.on_cluster}}),
        (ImageLevelQC, {
          'build': {
              'cell_seg_key': config.seg_key,
              'serum_key': serum_seg_in_key,
              'marker_key': marker_ana_in_key,
              'feature_identifier': feature_identifier,
              'outlier_predicate': outlier_predicate}})
    ])

    # for the background substraction, we can either use a fixed value per channel,
    # or compute it from the data. In the first case, we pass the value
    # as 'serum_bg_key' / 'marker_bg_key'. In the second case, we pass a key
    # to the table holding these values.
    background_parameters, bg_wells = parse_background_parameters(config, marker_ana_in_key, serum_ana_in_keys)

    # we could also just add these on demand depending on what we need according to the background params
    bg_estimation_keys = [marker_ana_in_key] + serum_ana_in_keys
    job_list = add_background_estimation(job_list, config.seg_key, bg_estimation_keys)
    if len(bg_wells) > 0:
        job_list = add_background_estimation_from_min_well(job_list, config, bg_wells, bg_estimation_keys)

    job_list = add_infected_detection_jobs(job_list, config, marker_ana_in_key, feature_identifier)

    table_path = CellLevelAnalysis.folder_to_table_path(config.folder)
    if config.write_background_images:
        job_list.append(
            (WriteBackgroundSubtractedImages, {
                 'build': {'background_dict': background_parameters,
                           'table_path': table_path,
                           'scale_factors': config.scale_factors},
                 'run': {'n_jobs': config.n_cpus,
                         'force_recompute': None}
            })
        )

    # we might add a second nucleus dilation task and feature extraction for image level intensity qc
    # at some point here, but not for the preprint

    table_identifiers = serum_ana_in_keys if feature_identifier is None else [k + f'_{feature_identifier}'
                                                                              for k in serum_ana_in_keys]
    write_summary_images = config.write_summary_images
    # NOTE currently the QC tasks will not be rerun if the feature identifier changes
    for serum_key, identifier in zip(serum_ana_in_keys, table_identifiers):

        image_outlier_table = f'images/outliers_{serum_key}'
        well_outlier_table = f'wells/outliers_{serum_key}'

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
                'table_out_key': image_outlier_table,
                'identifier': identifier}
        }))

        # if we have wells without serum, we use them to determine the min infected intensity
        # otherwise, we use the preset values determined on multiple plates
        # Note: the min intensity is set to 3 times the MAD
        well_qc_criteria = DEFAULT_WELL_OUTLIER_CRITERIA.copy()
        if len(bg_wells) > 0 and config.use_mad_from_bg_wells:
            min_infected_intensity_for_channel = 'plate/backgrounds_min_well'
        else:
            min_infected_intensity_for_channel = DEFAULT_MIN_SERUM_INTENSITIES.get(serum_key, None)
        well_qc_criteria.update({'min_infected_intensity': min_infected_intensity_for_channel})

        job_list.append((WellLevelQC, {
            'build': {
                'cell_seg_key': config.seg_key,
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'feature_identifier': feature_identifier,
                'outlier_criteria': well_qc_criteria,
                'table_out_key': well_outlier_table,
                'identifier': identifier},
            'run': {'force_recompute': None}
        }))
        job_list.append((CellLevelAnalysis, {
            'build': {
                'serum_key': serum_key,
                'marker_key': marker_ana_in_key,
                'cell_seg_key': config.seg_key,
                'serum_bg_key': background_parameters[serum_key],
                'marker_bg_key': background_parameters[marker_ana_in_key],
                'write_summary_images': write_summary_images,
                'scale_factors': config.scale_factors,
                'feature_identifier': feature_identifier,
                'image_outlier_table': image_outlier_table,
                'well_outlier_table': well_outlier_table,
                'identifier': identifier},
            'run': {'force_recompute': None}}))
        write_summary_images = False

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
                  'identifier': feature_identifier},
        'run': {'force_recompute': None}
    }))

    return job_list, table_identifiers, background_parameters, marker_ana_in_key


def bg_dict_for_plots(bg_params, table_path):
    bg_dict = {}
    with open_file(table_path, 'r') as f:
        for channel_name, bg_param in bg_params.items():
            if isinstance(bg_param, str):
                assert has_table(f, bg_param)
                cols, table = read_table(f, bg_param)
                bg_src = bg_param.split('/')[-1]
                if bg_param == 'plate/backgrounds':
                    bg_val = table[0, cols.index(f'{channel_name}_median')]
                    bg_msg = f' as {bg_val}'
                elif bg_param == 'plate/backgrounds_min_well':
                    bg_val = table[0, cols.index(f'{channel_name}_median')]
                    bg_wells = table[0, cols.index(f'{channel_name}_min_well')]
                    bg_src = f' the wells {bg_wells}'
                    bg_msg = f' as {bg_val}'
                else:
                    bg_msg = ''
                bg_info = f'background computed from {bg_src}{bg_msg}'
            else:
                bg_info = f'background fixed to {bg_param}'
            bg_dict[channel_name] = bg_info
    return bg_dict


def workflow_summaries(name, config, t0, workflow_name, input_folder, stat_names,
                       bg_params=None, marker_name='marker'):
    # run all plots on the output files
    plot_folder = os.path.join(config.folder, 'plots')

    table_name = 'default'
    table_path = CellLevelAnalysis.folder_to_table_path(config.folder)

    if bg_params is None:
        bg_dict = None
    else:
        bg_dict = bg_dict_for_plots(bg_params, table_path)

    all_plots(table_path, plot_folder,
              table_key=f'images/{table_name}',
              identifier='per-image',
              stat_names=stat_names,
              wedge_width=0.3,
              bg_dict=bg_dict)
    all_plots(table_path, plot_folder,
              table_key=f'wells/{table_name}',
              identifier='per-well',
              stat_names=stat_names,
              wedge_width=0,
              bg_dict=bg_dict)

    db_writer = DbResultWriter(
        workflow_name=workflow_name,
        plate_dir=input_folder,
        username=config.db_username,
        password=config.db_password,
        host=config.db_host,
        port=config.db_port,
        db_name=config.db_name
    )
    db_writer(config.folder, config.folder, t0=t0)

    t0 = time.time() - t0
    summary_writer = SlackSummaryWriter(config.slack_token)
    summary_writer(config.folder, config.folder, runtime=t0)

    if config.export_tables:
        export_tables_for_plate(config.folder, marker_name=marker_name)
    logger.info(f"Run {name} in {t0}s")


def run_cell_analysis(config):
    """
    """
    name = 'CellAnalysisWorkflow'
    feature_identifier = config.feature_identifier
    if feature_identifier is not None:
        name += f'_{feature_identifier}'

    job_list, table_identifiers, bg_params, marker_name = core_workflow_tasks(config, name, feature_identifier)

    t0 = time.time()
    run_workflow(name,
                 config.folder,
                 job_list,
                 input_folder=config.input_folder,
                 force_recompute=config.force_recompute,
                 ignore_invalid_inputs=config.ignore_invalid_inputs,
                 ignore_failed_outputs=config.ignore_failed_outputs,
                 skip_processed=bool(config.skip_processed))

    # only run the workflow summaries if we don't have the feature identifier
    if feature_identifier is None:
        stat_names = [idf.replace('serum_', '') + '_' + name
                      for name in DEFAULT_PLOT_NAMES for idf in table_identifiers]
        workflow_summaries(name, config, t0, workflow_name=name,
                           input_folder=config.input_folder, stat_names=stat_names,
                           bg_params=bg_params, marker_name=marker_name)

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
    parser.add('--barrel_corrector_folder', type=str, default='auto',
               help='optinally specify the folder containing the files for barrel correction.')

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
    parser.add("--ignore_nuclei", default=True)

    # parameter for the infected cell detection
    parser.add("--infected_erosion_radius", default=0, type=int)
    parser.add("--infected_tophat_radius", default=20, type=int)
    parser.add("--ignore_nuclei_in_infected_classification", default=False, type=bool)
    parser.add("--infected_scale_with_mad", default=True)
    parser.add("--infected_threshold", type=float, default=4.8)
    parser.add("--infected_quantile", type=float, default=0.95)

    # background subtraction values for the individual channels
    # if None, all backgrounds will be computed from the data and the plate background will be used
    parser.add("--background_dict", default=None)
    parser.add("--use_mad_from_bg_wells", default=True)

    # arguments for the nucleus dilation used for the serum intensity qc
    parser.add("--qc_dilation", type=int, default=5)

    #
    # more options
    #

    # runtime options
    parser.add("--batch_size", default=4)
    parser.add("--force_recompute", default=None)
    parser.add("--ignore_invalid_inputs", default=None)
    parser.add("--ignore_failed_outputs", default=None)
    parser.add("--skip_processed", default=0, type=int)

    # additional image output
    parser.add("--write_summary_images", default=True)
    parser.add("--write_background_images", default=True)

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
