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


def get_ann_and_tiff_files(config):
    # return gt_annotation files and corresponding raw tiffs
    ann_files = glob(os.path.join(config.ann_dir, 'infection/*/*infected_nuclei.h5'))
    tiff_files = list(map(partial(ann_to_in_file, config), ann_files))
    return ann_files, tiff_files


def ann_to_in_file(config, ann_file):
    plate = os.path.dirname(ann_file)
    if plate.endswith('_IgA') or plate.endswith('_IgG'):
        plate = plate[:-4]
    filename = os.path.basename(ann_file).rstrip('.h5') + '.tiff'
    filename = filename.replace('_infected_nuclei', '')
    #filename = filename.replace('_infected', '')
    in_file = os.path.join(config.data_dir, os.path.basename(plate), filename)
    return in_file


def in_to_out_file(config, in_file):
    plate = os.path.dirname(in_file)
    out_file = os.path.join(config.out_dir, os.path.basename(plate) + '_' +
                            os.path.basename(in_file).rstrip('.tiff') + '.h5')
    return out_file


def preprocess(config, ann_files, tiff_files):
    os.makedirs(config.out_dir, exist_ok=True)
    # group files by plate
    files_per_plate = defaultdict(list)
    for in_file in tiff_files:
        plate = os.path.dirname(in_file)
        files_per_plate[plate].append(in_file)

    # preprocess files by plate
    barrel_corrector_root = os.path.join(config.misc_folder, 'barrel_correctors')
    for plate, in_files in files_per_plate.items():
        print(plate)
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

    # copy the infected gt to the out_files

    out_files = list(map(partial(in_to_out_file, config), tiff_files))
    move_infected_gt = CopyImg('infected', 'infected')
    move_infected_gt.run(ann_files, out_files)


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
    def extract_feature_job_specification(seg_key, ignore_nuclei):
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
        return job_list[0]

    job_list = grid_evaluate(
        extract_feature_job_specification,
        seg_key=SearchSpace.segmentation_key,
        ignore_nuclei=SearchSpace.ignore_nuclei,
        n_jobs=0)['result'].reshape(-1).tolist()

    run_workflow('Feature Extraction Workflow',
                 config.out_dir,
                 job_list)


def find_infected(config, seg_key, ignore_nuclei, split_statistic, infected_threshold):
    identifier = get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)
    job_list = [((FindInfectedCells, {
        'build': {
            'marker_key': 'marker_corrected',
            'cell_seg_key': seg_key + '_' + get_identifier(seg_key, ignore_nuclei),
            # old method
            'bg_correction_key': 'image_bg_median',  # FIXME this should be the plate-wise bg. copy from results!
            'bg_cell_seg_key': 'cell_segmentation_cell_segmentation_False',  # weird because of identifier
            'split_statistic': split_statistic,
            'infected_threshold': infected_threshold,
            #'identifier': identifier,
            # new method
            # 'bg_correction_key': 'well_bg_median',
            'infected_threshold_scale_key': 'image_bg_mad',
            # 'infected_threshold': 7,
        }}))]


    #return job_list[0]
    run_workflow(f'Infected Classification Workflow '
                 f'({get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)})',
                 os.path.join(config.out_dir, identifier),
                 job_list,
                 config.out_dir,
                 force_recompute=True,
                 lock_folder=False,
                 enable_logging=False)


def find_infected_grid(config, SearchSpace):
    # TODO what about well wise / plate wise bgs? get them from somewhere!
    job_grid = grid_evaluate(partial(find_infected, config),
                  seg_key=SearchSpace.segmentation_key,
                  ignore_nuclei=SearchSpace.ignore_nuclei,
                  split_statistic=SearchSpace.split_statistic,
                  infected_threshold=SearchSpace.infected_threshold,
                  n_jobs=config.n_cpus)
    # job_list = job_grid['result'].reshape(-1).tolist()
    # print(len(job_list))
    # print(job_list[0])
    # run_workflow(f'Infected Classification Workflow ',
    #              config.out_dir,
    #              job_list,
    #              lock_folder=False)


class GetGTInfectedTable(BatchJobOnContainer):
    def __init__(self, nuc_seg_key, infected_key, out_key='infected', **super_kwargs):
        self.nuc_seg_key = nuc_seg_key
        self.infected_key = infected_key
        self.table_out_key = out_key
        super().__init__(
            input_key=[self.nuc_seg_key, self.infected_key],
            output_key='tables/' + out_key,
            **super_kwargs
        )

    def run(self, in_files, out_files):
        for in_file, out_file in zip(tqdm(in_files, desc='Extracting GT infected tables'), out_files):
            with open_file(in_file, 'r') as f:
                nuc_seg = self.read_image(f, self.nuc_seg_key)
                gt_img = self.read_image(f, self.infected_key)
            infected_mask = gt_img == 2
            labels = np.unique(nuc_seg)
            infected_indicator = np.array([i > 0 and np.mean(infected_mask[nuc_seg == i].astype(np.float32)) < 0.5
                                           for i in labels])
            control_indicator = np.array([i > 0 and np.mean(infected_mask[nuc_seg == i].astype(np.float32)) >= 0.5
                                          for i in labels])

            column_names = ['label_id', 'is_infected', 'is_control']
            table = [labels, infected_indicator, control_indicator]
            table = np.asarray(table, dtype=float).T
            with open_file(out_file, 'a') as f:
                self.write_table(f, self.table_out_key, column_names, table)


def save_gt_infected(config):
    job_list = [((GetGTInfectedTable, {
        'build': {
            'nuc_seg_key': config.nuc_key,
            'infected_key': 'infected',
        }}))]
    run_workflow(f'Infected Classification Workflow ',
                 config.out_dir,
                 job_list,
                 force_recompute=False)

def get_prediction_and_eval_score(
    config,
    seg_key,
    ignore_nuclei,
    split_statistic,
    infected_threshold,
    ann_file,
):
    identifier = get_identifier(seg_key, ignore_nuclei, split_statistic, infected_threshold)
    in_file = ann_to_in_file(config, ann_file)
    out_file = in_to_out_file(config, in_file)
    out_file_in_subdir = os.path.join(os.path.dirname(out_file), identifier, os.path.basename(out_file))
    assert os.path.isfile(out_file_in_subdir)
    assert os.path.isfile(out_file), f'Output file missing: {out_file}'
    with open_file(out_file_in_subdir, 'r') as f:
        column_names, table = read_table(f, 'cell_classification/' + seg_key + '_' +
                                         get_identifier(seg_key, ignore_nuclei) +
                                         '/marker_corrected')
    with open_file(out_file, 'r') as f:
        gt_column_names, gt_table = read_table(f, 'infected')
    infected_indicator = table[:, column_names.index('is_infected')]
    #control_indicator = table[:, column_names.index('is_control')]
    labels = table[:, column_names.index('label_id')].astype(np.int32)
    infected_dict = dict(zip(labels, infected_indicator))

    gt_infected_indicator = gt_table[:, gt_column_names.index('is_infected')]
    #gt_control_indicator = gt_table[:, gt_column_names.index('is_control')]
    gt_labels = gt_table[:, gt_column_names.index('label_id')].astype(np.int32)
    gt_infected_dict = dict(zip(gt_labels, gt_infected_indicator))

    accuracy = np.mean([infected_dict.get(i, 0) == gt_infected_dict.get(i, 0)
                        for i in set(labels).union(gt_labels) if i != 0])
    #n_cells = np.sum(labels != 0)
    return accuracy


def get_score_grid(config, SearchSpace, ann_files):
    print('computing scores')
    score_grid = grid_evaluate(
        partial(get_prediction_and_eval_score, config),
        seg_key=SearchSpace.segmentation_key,
        ignore_nuclei=SearchSpace.ignore_nuclei,
        split_statistic=SearchSpace.split_statistic,
        infected_threshold=SearchSpace.infected_threshold,
        ann_file=ann_files,
        n_jobs=config.n_cpus,
    )

    return score_grid


def run_grid_search_for_infected_cell_detection(config, SubParamRanges, SearchSpace):

    print('number of points on grid:', np.product([len(v) for v in [
        SearchSpace.segmentation_key,
        SearchSpace.ignore_nuclei,
        SearchSpace.split_statistic,
        SearchSpace.infected_threshold,
    ]]))


    ann_files, tiff_files = get_ann_and_tiff_files(config)
    print(f'Found input tiff files:')
    [print(f) for f in tiff_files]

    # if os.path.isdir(config.out_dir):
    #     shutil.rmtree(config.out_dir)

    preprocess(config, ann_files, tiff_files)

    compute_segmentations(config, SubParamRanges)

    extract_feature_grid(config, SubParamRanges, SearchSpace)

    save_gt_infected(config)

    find_infected_grid(config, SearchSpace)

    score_grid = get_score_grid(config, SearchSpace, tiff_files)
    print(score_grid['result'].max())

    np.save(os.path.join(config.out_dir, 'score_grid.npy'), score_grid)
