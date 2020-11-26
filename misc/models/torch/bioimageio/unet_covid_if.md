# UNetCovidIf

This model predicts cell membrane boundaries and foreground (= cellular space) based on immuno-fluorescence images of covid infeted vero cells.
It was developed for the semi-quantitative analysis of an immunofluorescence based SARS-CoV-2 antibody assay.
The results of this model are used in the cell segmentation procedure, that uses an additional nucleus segmentation as seeds for a watershed,
using the membrane boundary predictions as height-map and the thresholded foreground predictions as mask.

For details check out the [publication](https://www.biorxiv.org/content/biorxiv/early/2020/10/07/2020.06.15.152587.full.pdf).
