# Annotation with BigCat

We correct the cell segmentation with bigcat. See details on [installation](https://github.com/saalfeldlab/bigcat#install)
and [usage](https://github.com/saalfeldlab/bigcat/wiki/BigCat-User-Interface).
Note: installation via conda does not work (at least for me).
To install from source, you need java8.


## Usage

Convert input file to bigcat format (expects [this file format](https://github.com/hci-unihd/batchlib#data-model)) for
correcting the cell segmentation
```
python make_bigcat_cell_segmentation_project.py /path/to/input.h5 /path/to/output.h5
```
or correcting the infected cell detection:
```
python make_bigcat_infected_cells_project.py /path/to/input.h5 /path/to/output.h5
```

Start bigcat:
```
bigcat -i /path/to/output.h5 -r volumes/raw -l volumes/labels/fragments
```

Export bigcat segmentation result (only works after corrected segmentation was exported, see below):
```
python export_groundtruth.py /path/to/output.h5 /path/to/export.h5
```


## Useful bigcat functionality

- press `U` to mark segments as correct and hide them
- press `J` to toggle hiding marked segments
- press `ctrl + shift + S` to save the corrected segmentation
- hold `SPACE` to paint with currently selected id
- press `SHIFT + 0` to toggle raw data visibility
- press `SHIFT + 1` to toggle segmentation visibility
