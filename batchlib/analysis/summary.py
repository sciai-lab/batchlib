import os
import numpy as np

from ..base import BatchJobOnContainer
from ..util import (get_image_and_site_names, open_file,
                    write_table, write_image_information)


# TODO implement everything
class Summary(BatchJobOnContainer):
    """ Write summary of analysis per file.

    Produces the following output per image
    - mask image for cells classified as infected / non-infected
    - intensities for antibody channel (and serum channel?) per cell
    - summary information written to hdf5 attributes
    - table with all analysis scores and intermediate results
    """

    # TODO need to pass all relevant input / output keys
    def __init__(self, cell_seg_key='cell_segmentation',
                 input_pattern='*.h5', **super_kwargs):

        self.cell_seg_key = cell_seg_key
        input_ndim = [2]

        super().__init__(input_pattern=input_pattern, output_ext=None,
                         input_key=[cell_seg_key], input_ndim=input_ndim,
                         **super_kwargs)

    def write_summary_table(self, table_out_path):
        im_names, site_names = get_image_and_site_names(self.folder,
                                                        self.input_pattern)

        # just add two dummy columns for now
        column_names = ['score1', 'score2']
        column_dict = {name: [np.random.rand(), np.random.rand()] for name in site_names}

        write_table(self.folder, column_dict, column_names,
                    out_path=table_out_path,
                    pattern=self.input_pattern)

    def write_summary_information(self, path):
        image_info = 'dummy'
        well_info = 'dummy'
        write_image_information(path,
                                image_information=image_info,
                                well_information=well_info)

    def write_summary_image(self, in_path, out_path):
        pass

    # TODO use n_jobs to parallelize
    def run(self, input_files, output_files, n_jobs=1):

        # TODO
        # write images with per cell information:
        # - infected vs non-infected
        # - mean antibody response intensiry
        # - mean serum intensity (?)
        for in_path, out_path in zip(input_files, output_files):
            self.write_summary_image(in_path, out_path)

        # TODO
        # write a table with summary information for all images
        table_out_path = os.path.join(self.folder, 'analysis.csv')
        self.write_summary_table(table_out_path)

        # TODO
        # write summary information in the hdf5 tags per image
        for path in output_files:
            self.write_summary_information(path)

    # FIXME this is just a hack, remove once this is properly implemented!
    def check_output(self, path):
        return False
