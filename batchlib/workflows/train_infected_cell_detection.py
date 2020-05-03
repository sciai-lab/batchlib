import numpy as np
import os
import configargparse
from batchlib.base import BatchJobOnContainer
from functools import partial
from batchlib.util import open_file
from collections import defaultdict
from batchlib.preprocessing import get_barrel_corrector, get_serum_keys, Preprocess
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.voronoi_ring_segmentation import VoronoiRingSegmentation
from batchlib.segmentation.unet import UNet2D
from batchlib import run_workflow
from batchlib.analysis.cell_level_analysis import InstanceFeatureExtraction, FindInfectedCells

from tqdm.auto import tqdm
from glob import glob
import shutil

from ..util.grid_search import grid_evaluate


class DummyJob(BatchJobOnContainer):
    def run(self):
        pass


dummy_job = DummyJob()
read_table = dummy_job.read_table


class CopyImg(BatchJobOnContainer):
    def __init__(self, input_key, output_key, **super_kwargs):
        super(CopyImg, self).__init__(
            input_key=input_key,
            output_key=output_key,
            **super_kwargs
        )

    def run(self, in_files, out_files):
        for in_file, out_file in zip(tqdm(in_files, desc=f'copying {self.input_key} to {self.output_key}'), out_files):
            with open_file(in_file, 'r') as f:
                img = self.read_image(f, self.input_key)
            with open_file(out_file, 'a') as f:
                self.write_image(f, self.output_key, img)


def get_input_files(config):
    # for now, get some dummy files with which I can do everything up to evaluation
    plates = ['titration_plate_20200403_154849', '20200420_152417_316', 'plateK12rep1_20200430_155932_313']
    tiff_files = np.concatenate([np.random.choice(glob(os.path.join(config.data_dir, plate, '*.tiff')), 2)
                                 for plate in plates])
    return tiff_files


def in_to_out_file(config, in_file):
    plate = os.path.dirname(in_file)
    out_file = os.path.join(config.out_dir, os.path.basename(plate) + '_' +
                            os.path.basename(in_file).rstrip('.tiff') + '.h5')
    return out_file


def preprocess(config, tiff_files):
    os.makedirs(config.out_dir, exist_ok=True)
    # group files by plate
    files_per_plate = defaultdict(list)
    for in_file in tiff_files:
        plate = os.path.dirname(in_file)
        files_per_plate[plate].append(in_file)

    # preprocess files by plate
    barrel_corrector_root = os.path.join(config.misc_folder, 'barrel_correctors')
    for plate, in_files in files_per_plate.items():
        preprocess = Preprocess.from_folder(
            input_folder=plate,
            barrel_corrector_path=get_barrel_corrector(barrel_corrector_root, plate)
        )
        out_files = list(map(partial(in_to_out_file, config), in_files))
        preprocess.run(in_files, out_files)

        serum_keys = get_serum_keys(plate)
        try:
            serum_seg_in_key = next(iter(filter(lambda key: key.endswith('IgG'), serum_keys))) + '_corrected'
        except StopIteration:
            serum_seg_in_key = serum_keys[0] + '_corrected'
        if serum_seg_in_key != 'serum_corrected':
            rename_serum_key = CopyImg(serum_seg_in_key, 'serum_corrected')
            rename_serum_key.run(out_files, out_files)


def compute_segmentations(config, SubParamRanges):
    nuc_seg_in_key = 'nuclei_corrected'

    misc_folder = config.misc_folder

    torch_model_path = os.path.join(misc_folder, 'models/torch/fg_and_boundaries_V1.torch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    model_root = os.path.join(misc_folder, 'models/stardist')
    model_name = '2D_dsb2018'

    job_list = [
        (TorchPrediction, {
            'build': {
                'input_key': 'serum_corrected',
                'output_key': [config.mask_key, config.bd_key],
                'model_path': torch_model_path,
                'model_class': torch_model_class,
                'model_kwargs': torch_model_kwargs},
            'run': {
                'gpu_id': config.gpu,
                'batch_size': config.batch_size,
                'threshold_channels': {0: 0.5}}}),
        (StardistPrediction, {
            'build': {
                'model_root': model_root,
                'model_name': model_name,
                'input_key': nuc_seg_in_key,
                'output_key': config.nuc_key},
            'run': {
                'gpu_id': config.gpu,
                'n_jobs': config.n_cpus}}),
        (SeededWatershed, {
            'build': {
                'pmap_key': config.bd_key,
                'seed_key': config.nuc_key,
                'output_key': config.seg_key,
                'mask_key': config.mask_key},
            'run': {
                'erode_mask': 20,
                'dilate_seeds': 3,
                'n_jobs': config.n_cpus}})
    ]

    for r in SubParamRanges.ring_widths:
        job_list.append((VoronoiRingSegmentation, {
            'build': {
                'input_key': config.nuc_key,
                'output_key': f'voronoi_ring_segmentation{r}',
                'ring_width': r,
                'disk_not_rings': True,
            }
        }))

    run_workflow('Segmentation Workflow',
                 config.out_dir,
                 job_list,
                 force_recompute=False)


def get_identifier(*args):
    return '_'.join(map(str, args))


def extract_feature_grid(config, SubParamRanges, SearchSpace):
    # TODO: add denoise radius argument
    def extract_features(seg_key, ignore_nuclei):
        print('\n', seg_key, ignore_nuclei)
        job_list = [((InstanceFeatureExtraction, {
            'build': {
                'channel_keys': ['marker_corrected'],
                'nuc_seg_key_to_ignore': config.nuc_key if ignore_nuclei else None,
                'cell_seg_key': seg_key,
                'identifier': get_identifier(seg_key, ignore_nuclei),
                'topk': SubParamRanges.ks_for_topk,
                'quantiles': SubParamRanges.quantiles,
            },
            'run': {'gpu_id': config.gpu}}))]
        run_workflow('Feature Extraction Workflow',
                     config.out_dir,
                     job_list,
                     force_recompute=True)

    grid_evaluate(extract_features,
                  seg_key=SearchSpace.segmentation_key,
                  ignore_nuclei=SearchSpace.ignore_nuclei,
                  n_jobs=0)


def find_infected_grid(config, SearchSpace):
    def find_infected(seg_key, ignore_nuclei, split_statistic, infected_threshold):
        job_list = [((FindInfectedCells, {
            'build': {
                'marker_key': 'marker_corrected',
                'cell_seg_key': seg_key + '_' + get_identifier(seg_key, ignore_nuclei),
                # old method
                'bg_correction_key': 'image_bg_median',  #FIXME this is now referring to the current segmentatin (e.g. voronoi rings). should always be cell segmentation..
                'split_statistic': split_statistic,
                'infected_threshold': infected_threshold,
                'identifier': get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)
                # new method
                # 'bg_correction_key': 'well_bg_median',
                # 'infected_threshold_scale_key': 'well_bg_mad',
                # 'infected_threshold': 7,
            }}))]
        run_workflow(f'Infected Classification Workflow '
                     f'({get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)})',
                     config.out_dir,
                     job_list,
                     force_recompute=False)

    # TODO what about well wise / plate wise bgs? get them from somewhere!
    grid_evaluate(find_infected,
                  seg_key=SearchSpace.segmentation_key,
                  ignore_nuclei=SearchSpace.ignore_nuclei,
                  split_statistic=SearchSpace.split_statistic,
                  infected_threshold=SearchSpace.infected_threshold,
                  n_jobs=0)


# TODO use proper score function
def dummy_score(in_file, infected_indicator):
    return (np.mean(infected_indicator),
            np.sum(infected_indicator),
            len(infected_indicator) - np.sum(infected_indicator))


def get_score_grid(config, SearchSpace, in_files, score_func):
    def get_prediction_and_eval_score(
        score_func,
        seg_key,
        ignore_nuclei,
        split_statistic,
        infected_threshold
    ):
        identifier = get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)
        scores = []
        for in_file in in_files:
            out_file = in_to_out_file(config, in_file)
            assert os.path.isfile(out_file), f'Output file missing: {out_file}'
            with open_file(out_file, 'r') as f:
                column_names, table = read_table(f, 'cell_classification/' + seg_key + '_' +
                                                 get_identifier(seg_key, ignore_nuclei) +
                                                 '/marker_corrected_' + identifier)
            infected_indicator = table[:, column_names.index('is_infected')]
            control_indicator = table[:, column_names.index('is_control')]
            labels = table[:, column_names.index('label_id')].astype(np.int32)
            #n_cells = np.sum(labels != 0)
            scores.append(dummy_score(in_file, infected_indicator))
        return np.mean(scores, axis=0)

    score_grid = grid_evaluate(
        partial(get_prediction_and_eval_score, score_func=dummy_score),
        seg_key=SearchSpace.segmentation_key,
        ignore_nuclei=SearchSpace.ignore_nuclei,
        split_statistic=SearchSpace.split_statistic,
        infected_threshold=SearchSpace.infected_threshold,
        n_jobs=0
    )

    return score_grid


def run_grid_search_for_infected_cell_detection(config, SubParamRanges, SearchSpace):
    print('number of points on grid:', np.product([len(v) for v in [
        SearchSpace.segmentation_key,
        SearchSpace.ignore_nuclei,
        SearchSpace.split_statistic,
        SearchSpace.infected_threshold,
    ]]))

    if os.path.isdir(config.out_dir):
        shutil.rmtree(config.out_dir)

    tiff_files = get_input_files(config)
    print(f'Found input tiff files:')
    [print(f) for f in tiff_files]

    preprocess(config, tiff_files)

    compute_segmentations(config, SubParamRanges)

    extract_feature_grid(config, SubParamRanges, SearchSpace)

    find_infected_grid(config, SearchSpace)

    score_grid = get_score_grid(config, SearchSpace, tiff_files, even_split_score)
    print(score_grid)
    np.save(os.path.join(config.out_dir, 'score_grid.npy'), score_grid)





