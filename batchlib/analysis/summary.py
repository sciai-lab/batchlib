import os
from abc import ABC
from collections import defaultdict

import numpy as np
import pickle
from tqdm.auto import tqdm

from ..base import BatchJobOnContainer
from ..config import get_default_extension
from ..util import (get_image_and_site_names, get_logger,
                    open_file, seg_to_edges, read_table,
                    write_table, write_image_information,
                    image_name_to_well_name)

logger = get_logger('Workflow.BatchJob.Summary')


class Summary(BatchJobOnContainer, ABC):
    """ Base class for summary jobs.

    Deriving classes must implement
    - make_summary_table: return the analysis summary table
    - write_summary_images: write additional images
    - write_image_information: write image/well level analysis summary
    """
    def __init__(self, input_key, output_key,
                 input_ndim=None, output_ndim=None,
                 table_name=None, **super_kwargs):
        super().__init__(output_ext=None,
                         input_key=input_key, input_ndim=input_ndim,
                         output_key=output_key, output_ndim=output_ndim,
                         **super_kwargs)
        self.table_name = table_name

    @property
    def table_path(self):
        folder = self.folder
        plate_name = os.path.split(folder)[1]
        table_name = plate_name + '_analysis.csv' if self.table_name is None else self.table_name
        return os.path.join(folder, table_name)

    # in addition to the normal 'check_outputs', we make sure that the table exists
    def validate_outputs(self, output_files, folder, status, ignore_failed_outputs):
        if not os.path.isfile(self.table_path):
            msg = f'{self.name}: failed to compute table at {self.table_path}'
            if ignore_failed_outputs:
                logger.warning(msg)
            else:
                logger.error(msg)
                raise RuntimeError(msg)
        super().validate_outputs(output_files, folder, status, ignore_failed_outputs)

    def write_summary_table(self):
        column_dict, column_names = self.make_summary_table()

        # replace None with "NaN"
        column_dict = {site_name: [value if value is not None else 'NaN'
                                   for value in values]
                       for site_name, values in column_dict.items()}

        # if this table already exists, then extend the column_dict and the column_names
        if os.path.exists(self.table_path):
            old_column_dict, old_column_names = read_table(self.table_path)
            column_names = old_column_names + column_names
            column_dict = {site_name: old_column_dict[site_name] + column_dict[site_name]
                           for site_name in column_dict.keys()}

        logger.info(f'{self.name}: save analysis table to {self.table_path}')
        write_table(self.folder, column_dict, column_names,
                    out_path=self.table_path,
                    pattern=self.input_pattern)

    def run(self, input_files, output_files):

        # write summary images
        for in_path, out_path in zip(tqdm(input_files, desc='writing summary images'), output_files):
            self.write_summary_images(in_path, out_path)

        # write a table with summary information for all images
        self.write_summary_table()

        # write image summary information
        for path in output_files:
            self.write_summary_information(path)


class PixelLevelSummary(Summary):
    pass


class CellLevelSummary(Summary):
    """ Write summary of analysis per file.

    Produces the following output per image
    - mask image for cells classified as infected / non-infected
    - intensities for antibody channel (and serum channel?) per cell
    - summary information written to hdf5 attributes
    - table with all analysis scores and intermediate results
    """
    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 infected_cell_mask_key='infected_cell_mask',
                 serum_per_cell_mean_key='serum_per_cell_mean',
                 edge_key='cell_segmentation_edges',
                 score_key='ratio_of_median_of_means',
                 analysis_folder='instancewise_analysis_corrected',
                 outlier_predicate=lambda im: False,
                 table_name=None,
                 **super_kwargs):
        self.cell_seg_key = cell_seg_key
        self.serum_key = serum_key
        self.marker_key = marker_key

        # TODO we should switch to h5 tables and then don't need this any more
        self.analysis_folder = analysis_folder

        self.outlier_predicate = outlier_predicate
        self.score_key = score_key  # the main score for the table

        self.infected_cell_mask_key = infected_cell_mask_key
        self.serum_per_cell_mean_key = serum_per_cell_mean_key
        self.edge_key = edge_key

        self.outlier_predicate = outlier_predicate

        input_key = [cell_seg_key, serum_key, marker_key]
        input_ndim = [2, 2, 2]

        output_key = [infected_cell_mask_key, serum_per_cell_mean_key, edge_key]
        output_ndim = [2, 2, 2]

        super().__init__(input_key=input_key, input_ndim=input_ndim,
                         output_key=output_key, output_ndim=output_ndim,
                         table_name=table_name, **super_kwargs)

    def make_summary_table(self):
        im_names, site_names = get_image_and_site_names(self.folder,
                                                        self.input_pattern)

        # get per image results and statistics
        ext = get_default_extension()
        results = [self.load_result(os.path.join(self.folder, im_name + ext)) for im_name in im_names]
        measures = [result['measures'] for result in results]
        num_cells = [len(result['infected_ind']) for result in results]
        num_infected_cells = [np.sum(result['infected_ind']) for result in results]
        num_not_infected_cells = [total-infected for total, infected in zip(num_cells, num_infected_cells)]
        fraction_infected_cells = [infected / total for total, infected in zip(num_cells, num_infected_cells)]

        bg_inds = [np.argwhere(result['per_cell_statistics']['labels'] == 0)[0, 0]
                   if 0 in result['per_cell_statistics']['labels'] else -1
                   for result in results]
        img_size = results[0]['per_cell_statistics']['marker']['sizes'].sum()
        background_percentages = [result['per_cell_statistics']['marker']['sizes'][bg_ind] / img_size
                                  for result, bg_ind in zip(results, bg_inds)]

        cell_based_scores = [m[self.score_key] for m in measures]
        per_well_scores = defaultdict(list)
        for im_name, score in zip(im_names, cell_based_scores):
            if self.outlier_predicate(im_name):
                continue
            per_well_scores[image_name_to_well_name(im_name)].append(score)
        per_well_scores = {well: np.median(scores) for well, scores in per_well_scores.items()}
        per_well_scores = [per_well_scores.get(image_name_to_well_name(im_name), None) for im_name in im_names]

        column_names = ['cell_based_score',
                        'well_cell_based_score'
                        'num_cells',
                        'num_infected_cells',
                        'num_not_infected_cells',
                        'fraction_infected',
                        'background_percentage',
                        'outlier',
                        ] + list(measures[0].keys())

        column_dict = {site_name: [cell_based_scores[i] if not self.outlier_predicate(im_name) else None,
                                   per_well_scores[i],
                                   num_cells[i],
                                   num_infected_cells[i],
                                   num_not_infected_cells[i],
                                   fraction_infected_cells[i],
                                   background_percentages[i],
                                   self.outlier_predicate(im_name)
                                   ] + list(measures[i].values())
                       for i, (im_name, site_name) in enumerate(zip(im_names, site_names))}

        return column_dict, column_names

    # TODO write the proper infos here
    def write_summary_information(self, path):
        image_info = 'dummy'
        well_info = 'dummy'
        write_image_information(path,
                                image_information=image_info,
                                well_information=well_info)

    def load_result(self, in_path):
        ext = get_default_extension()
        assert in_path.endswith(ext)
        # load result of cell level analysis
        split_path = os.path.abspath(in_path).split(os.sep)
        result_path = os.path.join('/', *split_path[:-1], self.analysis_folder, split_path[-1][:-3] + '.pickle')
        assert os.path.isfile(result_path), f'Result file missing: {result_path}'
        with open(result_path, 'rb') as f:
            return pickle.load(f)

    def write_summary_images(self, in_path, out_path):
        result = self.load_result(in_path)
        labels = result['per_cell_statistics']['labels']
        labels = np.array([], dtype=np.int32) if labels is None else labels
        infected_labels = labels[result['infected_ind'] != 0]

        with open_file(in_path, 'r') as f:
            cell_seg = self.read_image(f, self.cell_seg_key)

        # TODO use np.isin instead
        infected_mask = np.zeros_like(cell_seg)
        for label in infected_labels:
            infected_mask[cell_seg == label] = 1

        mean_serum_image = np.zeros_like(cell_seg, dtype=np.float32)
        for label, intensity in zip(filter(lambda x: x != 0, labels), result['per_cell_statistics']['serum']['means']):
            mean_serum_image[cell_seg == label] = intensity

        seg_edges = seg_to_edges(cell_seg).astype('uint8')

        with open_file(in_path, 'a') as f:
            # we need to use nearest down-sampling for the mean serum images,
            # because while these are float values, they should not be interpolated
            self.write_image(f, self.serum_per_cell_mean_key, mean_serum_image,
                             settings={'use_nearest': True})
            self.write_image(f, self.infected_cell_mask_key, infected_mask)
            self.write_image(f, self.edge_key, seg_edges)
