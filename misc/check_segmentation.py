import h5py
import napari

from batchlib import run_workflow
from batchlib.preprocessing import Preprocess
from batchlib.segmentation.stardist_prediction import StardistPrediction
from batchlib.segmentation.seeded_watershed import SeededWatershed
from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D


def prepare():
    in_folder = '../../antibodies-data/seg-test'
    barrel_corrector_path = './barrel_corrector.h5'

    model_root = '../../antibodies-nuclei/stardist/models/pretrained'
    model_name = '2D_dsb2018'

    torch_model_path = '../../antibodies-nuclei/unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch'
    torch_model_class = UNet2D
    torch_model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }

    job_dict = {
        Preprocess.from_folder: {'build': {'input_folder': in_folder,
                                           'barrel_corrector_path': barrel_corrector_path}},
        TorchPrediction: {'build': {'input_key': 'serum_corrected',
                                    'output_key': ['mask', 'boundaries'],
                                    'model_path': torch_model_path,
                                    'model_class': torch_model_class,
                                    'model_kwargs': torch_model_kwargs},
                          'run': {'threshold_channels': {0: 0.5}}},
        StardistPrediction: {'build': {'model_root': model_root,
                                       'model_name': model_name,
                                       'input_key': 'nuclei',
                                       'output_key': 'seeds'}},
    }

    run_workflow('PrepareWS', 'seg-test/prepare', job_dict,
                 input_folder=in_folder)


def segment(ensure_seeds, erode_mask, dilate_seeds=3):
    out_key = 'seg/ensure=%s,erode=%i,dilate=%i' % ('True' if ensure_seeds else 'False',
                                                    erode_mask, dilate_seeds)

    job = SeededWatershed(pmap_key='boundaries',
                          seed_key='seeds',
                          mask_key='mask',
                          output_key=out_key)
    folder = 'seg-test/ws'
    job(folder, input_folder='seg-test/prepare',
        erode_mask=erode_mask, dilate_seeds=dilate_seeds, ensure_seeds=ensure_seeds)


def seg_tests():
    segment(False, 3)
    for erode in (7, 11, 15, 19):
        segment(True, erode)


def check_result():
    p = './seg-test/prepare/WellB04_PointB04_0008_ChannelDAPI,WF_GFP,TRITC_Seq0188.h5'
    with h5py.File(p, 'r') as f:
        raw = f['serum_corrected/s0'][:]

    p = './seg-test/ws/WellB04_PointB04_0008_ChannelDAPI,WF_GFP,TRITC_Seq0188.h5'
    with h5py.File(p, 'r') as f:
        g = f['seg']
        segs = {}
        for name in g:
            segs[name] = g[name + '/s0'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        for name, seg in segs.items():
            viewer.add_labels(seg, name=name)


def check_pre():
    p = './seg-test/prepare/WellB04_PointB04_0008_ChannelDAPI,WF_GFP,TRITC_Seq0188.h5'
    with h5py.File(p, 'r') as f:
        raw = f['serum_corrected/s0'][:]
        mask = f['mask/s0'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(mask, name='mask')


if __name__ == '__main__':
    prepare()
    # check_pre()
    seg_tests()
    check_result()
