from batchlib.workflows.train_infected_cell_detection import run_grid_search_for_infected_cell_detection
import numpy as np


if __name__ == '__main__':

    class SubParamRanges:
        ks_for_topk = [10, 30, 40, 45, 50, 55, 60, 100, 200]
        quantiles = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
        ring_widths = [20]  #[10, 50]  #[20, 50]

    class SearchSpace:
        segmentation_key = ['cell_segmentation'] + [f'voronoi_ring_segmentation{r}' for r in SubParamRanges.ring_widths]
        ignore_nuclei = [True, False]
        split_statistic = [f'top{k}' for k in SubParamRanges.ks_for_topk] + [f'quantile{q}' for q in
                                                                             SubParamRanges.quantiles]
        infected_threshold = np.arange(0, 20, 0.05)  #np.arange(0, 4000, 50)
        #marker_denoise_radii = [0, 5, 10]  # TODO use those

    # For debugging
    class SubParamRanges:
        ks_for_topk = [50]
        quantiles = [0.9]
        ring_widths = [5]

    class SearchSpace:
        segmentation_key = ['cell_segmentation'] + [f'voronoi_ring_segmentation{r}' for r in SubParamRanges.ring_widths]
        ignore_nuclei = [True, False]
        split_statistic = [f'top{k}' for k in SubParamRanges.ks_for_topk] + [f'quantile{q}' for q in
                                                                             SubParamRanges.quantiles]
        infected_threshold = np.arange(0, 20, 1)  #np.arange(0, 4000, 50)
        #marker_denoise_radii = [0, 5, 10]  # TODO use those

    class config:  # TODO make an argument parser for this
        ann_dir = '/home_sdc/rremme_tmp/src/antibodies-nuclei/groundtruth'
        data_dir = '/home_sdc/rremme_tmp/DatasetsHCIHome/antibodies/covid-data-vibor/'
        out_dir = '/home_sdc/rremme_tmp/DatasetsHCIHome/antibodies/grid_search/debug'
        misc_folder = '/home_sdc/rremme_tmp/src/batchlib/misc'

        mask_key = 'mask'
        bd_key = 'boundaries'
        nuc_key = 'nuclei_segmentation'
        seg_key = 'cell_segmentation'
        gpu = 0
        batch_size = 1
        n_cpus = 20

    print(SearchSpace.__dict__)

    run_grid_search_for_infected_cell_detection(config, SubParamRanges, SearchSpace)
