from batchlib.workflows.train_infected_cell_detection import run_grid_search_for_infected_cell_detection
import numpy as np


if __name__ == '__main__':

    class SubParamRanges:
        ks_for_topk = [40, 45, 50, 55, 60]
        quantiles = [0.8, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
        ring_widths = [10, 20]  #[10, 50]  #[20, 50]
        segmentation_erode_radii = [1, 2, 3, 5, 10]

    class SearchSpace:
        segmentation_key = ['cell_segmentation'] + \
                           [f'eroded_cell_segmentation{r}' for r in SubParamRanges.segmentation_erode_radii] + \
                           [f'voronoi_ring_segmentation{r}' for r in SubParamRanges.ring_widths]
        ignore_nuclei = [True, False]
        split_statistic = [f'top{k}' for k in SubParamRanges.ks_for_topk] + [f'quantile{q}' for q in
                                                                             SubParamRanges.quantiles]
        infected_threshold = np.arange(0, 3000, 25)  #np.arange(0, 4000, 50)
        marker_denoise_radii = [0, 1, 2, 5, 10]

    # For debugging
    # class SubParamRanges:
    #     ks_for_topk = [50]
    #     quantiles = [0.9, 0.97]
    #     ring_widths = []
    #     segmentation_erode_radii = [1, 2]
    #
    # class SearchSpace:
    #     segmentation_key = ['cell_segmentation'] + \
    #                        [f'eroded_cell_segmentation{r}' for r in SubParamRanges.segmentation_erode_radii] + \
    #                        [f'voronoi_ring_segmentation{r}' for r in SubParamRanges.ring_widths]
    #
    #     ignore_nuclei = [True]
    #     split_statistic = [f'top{k}' for k in SubParamRanges.ks_for_topk] + [f'quantile{q}' for q in
    #                                                                          SubParamRanges.quantiles]
    #     infected_threshold = np.arange(0, 5000, 500)
    #     marker_denoise_radii = [0, 5]

    class config:  # TODO make an argument parser for this
        ann_dir = '/home_sdc/rremme_tmp/src/antibodies-nuclei/groundtruth'
        data_dir = '/home_sdc/rremme_tmp/DatasetsHCIHome/antibodies/covid-data-vibor/'
        out_dir = '/home_sdc/rremme_tmp/Datasets/covid_antibodies/grid_search_20200510_02'
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
