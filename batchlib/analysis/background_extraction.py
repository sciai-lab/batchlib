import os
from collections import defaultdict

import numpy as np
import torch

from .cell_level_analysis import CellLevelAnalysisWithTableBase, _load_image_outliers
from ..base import BatchJobOnContainer
from ..util import get_logger
from ..util.io import (open_file, image_name_to_well_name,
                       in_file_to_image_name, in_file_to_plate_name,
                       add_site_name_to_image_table)

logger = get_logger('Workflow.BatchJob.ExtractBackground')


class BackgroundFromMinWell(BatchJobOnContainer):
    def __init__(self, bg_table, output_table,
                 channel_names, min_background_fraction):

        self.bg_table = bg_table
        self.output_table = output_table

        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=self.bg_table,
                         input_format='table',
                         output_key=self.output_table, output_format='table')

        self.min_background_fraction = min_background_fraction
        self.channel_names = channel_names

    def run(self, input_files, output_files):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        in_file, out_file = input_files[0], output_files[0]
        with open_file(in_file, 'r') as f:
            col_names, table = self.read_table(f, self.bg_table)

        # get the background fraction and find wells that don't
        # have enough background
        bg_fraction = table[:, col_names.index('background_fraction')]
        well_names = table[:, col_names.index('well_name')]
        invalid_wells = bg_fraction < self.min_background_fraction

        logger.info(f"{self.name}: {invalid_wells.sum()} wells will not be considered for the min background")
        logger.info(f"{self.name}: because they have a smaller background fraction than {self.min_background_fraction}")
        logger.debug(f"{self.name}: the following wells are invalid {well_names[invalid_wells]}")

        plate_name = os.path.split(self.folder)[1]

        out_col_names = ['plate_name']
        out_table = [plate_name]
        for channel_name in self.channel_names:
            median_col_name = f'{channel_name}_median'
            mad_col_name = f'{channel_name}_mad'

            medians = table[:, col_names.find(median_col_name)]
            mads = table[:, col_names.find(mad_col_name)]

            medians[invalid_wells] = np.inf
            min_id = np.argmin(medians)

            min_well, min_median, min_mad = well_names[min_id], medians[min_id], mads[min_id]

            out_table.extend([min_well, min_median, min_mad])
            out_col_names.extend([f'{channel_name}_min_well', median_col_name, mad_col_name])

        out_table = np.array(table)[None]
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_table, out_col_names, table)


# TODO need to compute the background fraction and save in column 'background_fraction'
class ExtractBackground(CellLevelAnalysisWithTableBase):
    def __init__(self,
                 cell_seg_key, serum_key, marker_key,  # TODO: get rid of this in CellLevelAnalysisWithTableBase
                 actual_channels_to_use,
                 image_outlier_table='images/outliers',
                 identifier=None,
                 **super_kwargs):

        self.channel_keys = actual_channels_to_use
        self.image_outlier_table = image_outlier_table

        group_name = 'backgrounds' if identifier is None else f'backgrounds_{identifier}'
        self.image_table_key = 'images/' + group_name
        self.well_table_key = 'wells/' + group_name
        self.plate_table_key = 'plate/' + group_name

        super().__init__(
            cell_seg_key=cell_seg_key, serum_key=serum_key, marker_key=marker_key,
            table_out_keys=[self.image_table_key, self.well_table_key, self.plate_table_key],
            validate_cell_classification=False,
            identifier=identifier,
            **super_kwargs
        )

    def get_bg_segment(self, path, device):
        with open_file(path, 'r') as f:
            channels = [self.read_image(f, key) for key in self.channel_keys]
            cell_seg = self.read_image(f, self.cell_seg_key)

        channels = [torch.FloatTensor(channel.astype(np.float32)).to(device) for channel in channels]
        cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)

        return torch.stack(channels)[:, cell_seg == 0]

    def get_bg_stats(self, bg_values):
        if bg_values is None:
            return {'median': [np.nan] * len(self.channel_keys),
                    'mad': [np.nan] * len(self.channel_keys)}
        # bg_vales should have shape n_channels, n_pixels
        bg_values = bg_values.cpu().numpy().astype(np.float32)
        medians = np.median(bg_values, axis=1)
        mads = np.median(np.abs(bg_values - medians[:, None]), axis=1)

        return {'median': medians, 'mad': mads}

    def stat_dict_to_table(self, stat_dict, first_column_name):
        column_names = [first_column_name] + [f'{channel}_{stat}'
                                              for channel in self.channel_keys
                                              for stat in list(next(iter(stat_dict.values())))]
        table = [list(stat_dict.keys())] + [[d[stat][i] for d in stat_dict.values()]
                                            for i, _ in enumerate(self.channel_keys)
                                            for stat in list(next(iter(stat_dict.values())))]
        table = np.array(table).T
        if first_column_name == 'image_name':
            # image tables need an extra 'site_name' column
            column_names, table = add_site_name_to_image_table(column_names, table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]
        return column_names, table

    def write_stat_dict_to_talbe(self, stat_dict, first_column_name, table_key):
        column_names, table = self.stat_dict_to_table(stat_dict, first_column_name)
        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, table_key, column_names, table)

    def run(self, input_files, out_files):
        # first, get plate wide and per-well background statistics
        logger.info(f'{self.name}: computing background statistics')
        bg_dict = {file: self.get_bg_segment(file, device='cpu') for file in input_files}
        bg_per_image_stats = {in_file_to_image_name(file): self.get_bg_stats(bg_segments)
                              for file, bg_segments in bg_dict.items()}

        # ignore image outliers in per_well and per_plate backgrounds
        outlier_dict = _load_image_outliers(self.name, self.table_out_path, self.image_outlier_table, input_files)
        bg_per_well_dict = defaultdict(list)
        wells = set()
        for file, bg_segment in bg_dict.items():
            well = image_name_to_well_name(os.path.basename(file))
            wells.add(well)
            image_name = os.path.splitext(os.path.split(file)[1])[0]
            if outlier_dict.get(image_name, (-1, None))[0] == 1:
                logger.info(f'{self.name}: skipping outlier image in bg calculation')
                continue
            bg_per_well_dict[well].append(bg_segment)
        bg_per_well_dict = {well: torch.cat(bg_segments, dim=1) for well, bg_segments in bg_per_well_dict.items()}
        bg_per_well_stats = {well: self.get_bg_stats(bg_per_well_dict.get(well, None)) for well in wells}

        bg_plate_stats = self.get_bg_stats(torch.cat(list(bg_per_well_dict.values()), dim=1)
                                           if len(bg_per_well_dict) > 0 else None)

        # save the results in tables
        self.write_stat_dict_to_talbe(bg_per_image_stats, 'image_name', self.image_table_key)
        self.write_stat_dict_to_talbe(bg_per_well_stats, 'well_name', self.well_table_key)
        plate_name = in_file_to_plate_name(input_files[0])
        self.write_stat_dict_to_talbe({plate_name: bg_plate_stats}, 'plate_name', self.plate_table_key)
