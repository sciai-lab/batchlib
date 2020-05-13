# U-Net for fg/bg and boundary prediction

## Getting started
Uses [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) to train a 2D U-Net on the task of predicting foreground and background
as well as boundaries between cells in the serum channel.

Install with conda:
```
conda create -n 3dunet -c conda-forge -c awolny pytorch-3dunet
```

## Train
Make sure to change `loaders/train/file_paths` as well as `loaders/val/file_paths` in the training YAML config
to point to your training and validation stacks. Also change `trainer/checkpoint_dir` to point to dir
where you want to save the model. 

Train the network to predict foreground and boundaries (foreground in the 1st channel, boundary in the 2nd channel):
```
train3dunet --config config/config_train_fg_and_boundaries.yml
```

## Predict
See [torch_prediction.py](batchlib/segmentation/torch_prediction.py)