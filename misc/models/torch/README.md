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

IMPORTANT: `pytorch-3dunet` package was designed for volumetric data and even when training with 2D images it requires
a singleton z-dimension to be present, i.e. convert your 2D `XY` raw/label datasets to `1XY` or for multi-channel
raw input `CXY`, convert it to `C1XY`.  

Train the network to predict foreground and boundaries (foreground in the 1st channel, boundary in the 2nd channel):
```
train3dunet --config config/config_train_fg_and_boundaries.yml
```

## Pre-trained networks:

- [fg_and_boundaries_V1.torch](fg_and_boundaries_V1.torch) - network trained on 4 initial GT files (3 for training, 1 for validation)
- [fg_and_boundaries_V2.torch](fg_and_boundaries_V2.torch) - network trained on 10 GT files from `antibodies-nuclei` rep. 
All 10 files were used for training (network was trained for 30K iterations). Standardization `((x - mean) / std)` 
was used to normalize the input where `mean`/`std` were computed on all of raw datasets part from the ones containing the large bright object artifacts. 

## Predict
See [torch_prediction.py](batchlib/segmentation/torch_prediction.py)