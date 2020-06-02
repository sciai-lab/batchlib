import argparse
import json
import os

import h5py
import numpy as np
import yaml

from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.util import read_image
from batchlib.workflow import run_workflow

from seg_metrics import MeanAveragePrecision


def segment(torch_model_path, data_path, cv_id, gpu=None, n_cpus=1):
    serum_key = 'serum_IgG'
    mask_key = 'for_eval_{cv_id}/mask'
    boundary_key = f'for_eval_{cv_id}/boundaries'
    nuc_in_key = 'nuclei'
    nuc_key = f'for_eval_{cv_id}/nucleus_segmentation'
    seg_key = f'for_eval_{cv_id}/cell_segmentation'

    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    model_root = os.path.join('../models/stardist')
    model_name = '2D_dsb2018'

    jobs = [
        (TorchPrediction, {
            'build': {
                'input_key': serum_key,
                'output_key': [mask_key, boundary_key],
                'model_path': torch_model_path,
                'model_class': torch_model_class,
                'model_kwargs': torch_model_kwargs},
            'run': {
                'gpu_id': gpu,
                'batch_size': 1,
                'threshold_channels': {0: 0.5},
                'on_cluster': False}}),
        (StardistPrediction, {
            'build': {
                'model_root': model_root,
                'model_name': model_name,
                'input_key': nuc_in_key,
                'output_key': nuc_key},
            'run': {
                'gpu_id': gpu,
                'n_jobs': n_cpus,
                'on_cluster': False}}),
        (SeededWatershed, {
            'build': {
                'pmap_key': boundary_key,
                'seed_key': nuc_key,
                'output_key': seg_key,
                'mask_key': mask_key},
            'run': {
                'erode_mask': 20,
                'dilate_seeds': 3,
                'n_jobs': n_cpus}}),
    ]
    folder = os.path.split(data_path)[0]
    run_workflow('SegmentForEval', folder, jobs, force_recompute=False)
    return seg_key


def evaluate(seg_path, seg_key, gt_path, gt_key, check_visually=False, average=False):
    metrics = MeanAveragePrecision()
    with h5py.File(seg_path, 'r') as f:
        seg = read_image(f, seg_key)
    with h5py.File(gt_path, 'r') as f:
        gt = read_image(f, gt_key)

    res = metrics(seg, gt, average=average)

    if check_visually:
        import napari
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_labels(seg, name='segmentation')
            viewer.add_labels(gt, name='ground-truth')
    return res


def evaluate_seg(torch_model_path, data_path, gt_path, cv_id, gpu=None, n_cpus=4):
    seg_key = segment(torch_model_path, data_path, cv_id, gpu, n_cpus)
    gt_key = 'cells'
    eval_res = evaluate(data_path, seg_key, gt_path, gt_key)
    return eval_res


def get_val_path(config):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    val_loader = config['loaders']['val']
    paths = val_loader['file_paths']
    assert len(paths) == 1
    val_path = paths[0]
    return val_path


def get_data_path(data_folder, val_path):
    folder2, file_name = os.path.split(val_path)
    plate_name = os.path.split(folder2)[1]
    file_name = file_name.replace('_segmentation_done', '')
    data_path = os.path.join(data_folder, plate_name, file_name)
    return data_path


def get_gt_path(gt_folder, val_path):
    folder2, file_name = os.path.split(val_path)
    plate_name = os.path.split(folder2)[1]
    gt_path = os.path.join(gt_folder, plate_name, file_name)
    return gt_path


def evaluation_cross_validation(misc_folder):
    data_folder = os.path.join(misc_folder, 'groundtruth/data')
    gt_folder = os.path.join(misc_folder, 'groundtruth/segmentation')
    model_root = os.path.join(misc_folder, 'unet_segmentation')

    results = {}
    for cv_id in range(1, 11):
        print("Run validation for", cv_id)
        model_folder = os.path.join(model_root, f'lou_config{cv_id}')
        model_path = os.path.join(model_folder, 'best_checkpoint.pytorch')
        assert os.path.exists(model_path)
        config = os.path.join(model_folder, f'train_fg_and_boundaries{cv_id}.yml')
        val_path = get_val_path(config)
        data_path = get_data_path(data_folder, val_path)
        if not os.path.exists(data_path):
            print("Could not find", data_path)
            continue

        gt_path = get_gt_path(gt_folder, val_path)
        assert os.path.exists(gt_path)
        res = evaluate_seg(model_path, data_path, gt_path, cv_id)
        results[cv_id] = res

    with open('./seg_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)


def find_diff(misc_folder):
    data_folder = os.path.join(misc_folder, 'groundtruth/segmentation')
    model_root = os.path.join(misc_folder, 'unet_segmentation')

    data_paths_exp = []
    for cv_id in range(1, 11):
        config = os.path.join(model_root, f'lou_config{cv_id}',
                              f'train_fg_and_boundaries{cv_id}.yml')
        val_path = get_val_path(config)
        data_path = get_gt_path(data_folder, val_path)
        data_paths_exp.append(data_path)

    data_paths = []
    for root, dirs, files in os.walk(data_folder):
        for name in files:
            if os.path.isdir(name):
                continue
            data_paths.append(os.path.join(root, name))

    data_paths = set(data_paths)
    data_paths_exp = set(data_paths_exp)

    unmatched = list(data_paths - data_paths_exp)
    print("Missing:")
    print(list(data_paths_exp - data_paths))

    import napari
    for pp in unmatched:
        with napari.gui_qt():
            print(pp)
            with h5py.File(pp, 'r') as f:
                raw = read_image(f, 'raw')
                seg = read_image(f, 'cells')
            viewer = napari.Viewer(title=pp)
            viewer.add_image(raw, name='raw')
            viewer.add_labels(seg, name='seg')


def mean_accuracy():
    with open('./seg_eval_results.json') as f:
        res = json.load(f).values()
    res = [re['0.5'] for re in res]
    print(np.mean(res), "+-", np.std(res))


if __name__ == '__main__':
    default_misc = '/home/pape/Work/covid/antibodies-nuclei'
    parser = argparse.ArgumentParser()
    parser.add_argument('--misc_folder', type=str, default=default_misc)
    args = parser.parse_args()

    mean_accuracy()
    # find_diff(args.misc_folder)
    # evaluation_cross_validation(args.misc_folder)
