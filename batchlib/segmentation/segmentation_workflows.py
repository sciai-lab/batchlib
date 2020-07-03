import os

from batchlib.analysis.feature_extraction import SegmentationProperties
from batchlib.segmentation import SeededWatershed
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D


# TODO model paths as optional parameter
def watershed_segmentation_workflow(config, seg_in_key, nuc_in_key, job_list,
                                    erode_mask=20, dilate_seeds=3, threshold=.5,
                                    min_nucleus_size=None):

    model_root = os.path.join(config.misc_folder, 'models/stardist')
    model_name = '2D_dsb2018'

    torch_model_path = os.path.join(config.misc_folder, 'models/torch/fg_and_boundaries_V2.torch')
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    job_list.extend([
        (TorchPrediction, {
            'build': {
                'input_key': seg_in_key,
                'output_key': [config.mask_key, config.bd_key],
                'model_path': torch_model_path,
                'model_class': torch_model_class,
                'model_kwargs': torch_model_kwargs,
                'scale_factors': config.scale_factors},
            'run': {
                'gpu_id': config.gpu,
                'batch_size': config.batch_size,
                'threshold_channels': {0: threshold},
                'on_cluster': config.on_cluster}}),
        (StardistPrediction, {
            'build': {
                'model_root': model_root,
                'model_name': model_name,
                'input_key': nuc_in_key,
                'output_key': config.nuc_key,
                'scale_factors': config.scale_factors},
            'run': {
                'gpu_id': config.gpu,
                'n_jobs': config.n_cpus,
                'on_cluster': config.on_cluster,
                'min_size': min_nucleus_size}}),
        (SeededWatershed, {
            'build': {
                'pmap_key': config.bd_key,
                'seed_key': config.nuc_key,
                'output_key': config.seg_key,
                'mask_key': config.mask_key,
                'scale_factors': config.scale_factors},
            'run': {
                'erode_mask': erode_mask,
                'dilate_seeds': dilate_seeds,
                'n_jobs': config.n_cpus}}),
        (SegmentationProperties, {
            'build': {'seg_key': config.seg_key},
            'run': {'n_jobs': config.n_cpus}})
    ])
    return job_list
