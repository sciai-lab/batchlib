import os
import numpy as np
import pickle
from tqdm.auto import tqdm

from ..base import BatchJobOnContainer
from ..util import (get_image_and_site_names, open_file,
                    write_table, write_image_information)


class Summary(BatchJobOnContainer):
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
                 score_key='ratio_of_median_of_means',
                 analysis_folder='instancewise_analysis_corrected',
                 input_pattern='*.h5',
                 outlier_predicate=lambda im: False,
                 **super_kwargs):
        self.cell_seg_key = cell_seg_key
        self.serum_key = serum_key
        self.marker_key = marker_key
        self.analysis_folder = analysis_folder

        self.score_key = score_key  # the main score for the table

        self.infected_cell_mask_key = infected_cell_mask_key
        self.serum_per_cell_mean_key = serum_per_cell_mean_key
        self.outlier_predicate = outlier_predicate

        input_ndim = [2]

        super().__init__(input_pattern=input_pattern, output_ext=None,
                         input_key=[cell_seg_key], input_ndim=input_ndim,
                         output_key=[infected_cell_mask_key, serum_per_cell_mean_key],
                         **super_kwargs)

    def write_summary_table(self, table_out_path):
        im_names, site_names = get_image_and_site_names(self.folder,
                                                        self.input_pattern)

        # get per image results and statistics
        results = [self.load_result(os.path.join(self.folder, im_name + '.h5')) for im_name in im_names]
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

        column_names = ['score1',
                        'num_cells',
                        'num_infected_cells',
                        'num_not_infected_cells',
                        'fraction_infected',
                        'background_percentage',
                        'outlier',
                        ] + list(measures[0].keys())

        column_dict = {site_name: [measures[i][self.score_key],
                                   num_cells[i],
                                   num_infected_cells[i],
                                   num_not_infected_cells[i],
                                   background_percentages[i],
                                   fraction_infected_cells[i],
                                   background_percentages[i],
                                   self.outlier_predicate(im_name)
                                   ] + list(measures[i].values())
                       for i, (im_name, site_name) in enumerate(zip(im_names, site_names))}
        # replace None with "NaN"
        column_dict = {site_name: [value if value is not None else 'NaN'
                                   for value in values]
                       for site_name, values in column_dict.items()}

        write_table(self.folder, column_dict, column_names,
                    out_path=table_out_path,
                    pattern=self.input_pattern)

    def write_summary_information(self, path):
        image_info = 'dummy'
        well_info = 'dummy'
        write_image_information(path,
                                image_information=image_info,
                                well_information=well_info)

    def load_result(self, in_path):
        assert in_path.endswith('.h5')
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
            cell_seg = self.read_input(f, self.cell_seg_key)

        infected_mask = np.zeros_like(cell_seg)
        for label in infected_labels:
            infected_mask[cell_seg == label] = 1

        mean_serum_image = np.zeros_like(cell_seg, dtype=np.float32)
        for label, intensity in zip(filter(lambda x: x != 0, labels), result['per_cell_statistics']['serum']['means']):
            mean_serum_image[cell_seg == label] = intensity

        with open_file(in_path, 'a') as f:
            self.write_result(f, self.infected_cell_mask_key, infected_mask)
            self.write_result(f, self.serum_per_cell_mean_key, mean_serum_image)

    # TODO use n_jobs to parallelize
    def run(self, input_files, output_files, n_jobs=1):
        # write images with per cell information:
        # - mean serum intensity (?)
        for in_path, out_path in zip(tqdm(input_files, desc='writing summary images'), output_files):
            self.write_summary_images(in_path, out_path)

        # write a table with summary information for all images
        table_out_path = os.path.join(self.folder, 'analysis.csv')
        self.write_summary_table(table_out_path)

        # TODO write summary information in the hdf5 tags per image
        for path in output_files:
            self.write_summary_information(path)

    def check_output(self, path):
        if not os.path.isfile(os.path.join(os.path.dirname(path), 'analysis.csv')):
            return False
        return super(Summary, self).check_output(path)

    def check_outputs(self, output_files, folder, status, ignore_failed_outputs):
        # TODO check summary table?
        return super(Summary, self).check_outputs(output_files, folder, status, ignore_failed_outputs)
