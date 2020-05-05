from batchlib.workflows.train_infected_cell_detection import run_grid_search_for_infected_cell_detection
import numpy as np


if __name__ == '__main__':

    class SubParamRanges:
        ks_for_topk = [10, 30, 50, 100, 200]
        quantiles = [0.9, 0.99]
        ring_widths = [10, 50]  #[20, 50] # fix bg first, before this makes no sense

    class SearchSpace:
        segmentation_key = ['cell_segmentation'] + [f'voronoi_ring_segmentation{r}' for r in SubParamRanges.ring_widths]
        ignore_nuclei = [True, False]
        split_statistic = ['means'] + [f'top{k}' for k in SubParamRanges.ks_for_topk] + [f'quantile{q}' for q in
                                                                                         SubParamRanges.quantiles]
        infected_threshold = np.arange(0, 4000, 50)
        marker_denoise_radii = [0, 5, 10]  # TODO use those

    class config:  # TODO make an argument parser for this
        ann_dir = '/home_sdc/rremme_tmp/src/antibodies-nuclei/groundtruth'
        data_dir = '/home_sdc/rremme_tmp/DatasetsHCIHome/antibodies/covid-data-vibor/'
        out_dir = '/home_sdc/rremme_tmp/Datasets/covid_antibodies/grid_search'
        misc_folder = '/home_sdc/rremme_tmp/src/batchlib/misc'

        mask_key = 'mask'
        bd_key = 'boundaries'
        nuc_key = 'nuclei_segmentation'
        seg_key = 'cell_segmentation'
        gpu = 0
        batch_size = 1
        n_cpus = 10

    run_grid_search_for_infected_cell_detection(config, SubParamRanges, SearchSpace)
