from batchlib.segmentation.torch_prediction import TorchPrediction
from batchlib.segmentation.unet import UNet2D
from batchlib.util.image import standardize

if __name__ == '__main__':
    model_path = '/home/adrian/workspace/antibodies-nuclei/unet_segmentation/sample_models/fg_boundaries_best_checkpoint.pytorch'
    model_class = UNet2D
    model_kwargs = {
        'in_channels': 1,
        'out_channels': 2,
        'f_maps': [32, 64, 128, 256, 512],
        'testing': True
    }
    tp = TorchPrediction('raw', ['foreground', 'boundaries'], model_path, model_class, model_kwargs)

    input_files = [['/home/adrian/workspace/antibodies-nuclei/groundtruth/WellC05_PointC05_0004_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0256.h5']]
    output_files = [['/home/adrian/workspace/antibodies-nuclei/groundtruth/WellC05_PointC05_0004_ChannelDAPI,WF_GFP,TRITC,WF_Cy5_Seq0256_pred.h5']]
    tp.run(input_files=input_files, output_files=output_files, batch_size=1, normalize=standardize, gpu_id='0')
