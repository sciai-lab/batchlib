import os
from copy import copy, deepcopy
from collections import defaultdict
from concurrent import futures

import numpy as np
import skimage.morphology
from scipy.stats import ttest_ind
from tqdm.auto import tqdm
from glob import glob

from batchlib.util.logging import get_logger
from ..base import BatchJobOnContainer
from ..util.image import seg_to_edges
from ..util.io import (open_file, image_name_to_site_name, image_name_to_well_name, get_column_dict,
                       has_table, read_table, write_image_information,
                       in_file_to_image_name, to_image_table)

logger = get_logger('Workflow.BatchJob.CellLevelAnalysis')


def index_cell_properties(cell_properties, ind):
    return {key: {inner_key: inner_value[ind.astype(np.bool)]
                  for inner_key, inner_value in value.items()} if isinstance(value, dict) else None
            for key, value in cell_properties.items()}


def join_cell_properties(*cell_property_list):
    # copy to avoid changing inputs
    cell_property_list = list(map(copy, cell_property_list))
    # remove labels if present as they are meaningless without info on what image the cell belongs to
    for cell_properties in cell_property_list:
        cell_properties.pop('labels', None)
    return {channel: {inner_key: np.concatenate([cell_property[channel][inner_key]
                                                 for cell_property in cell_property_list
                                                 if len(cell_property[channel][inner_key].shape) > 0])
                      for inner_key in per_channel_properties.keys()}
            for channel, per_channel_properties in cell_property_list[0].items()}


def nan_on_exception(func):
    def inner(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            result = np.nan
        return result
    return inner


def compute_global_statistics(cell_properties):
    result = dict()
    for channel, properties in cell_properties.items():
        if not isinstance(properties, dict):
            continue
        result[channel] = dict()
        result[channel]['n_pixels'] = properties['sizes'].sum()
        result[channel]['sum_over_pixels'] = properties['sums'].sum()
        result[channel]['mean_over_pixels'] = result[channel]['sum_over_pixels'] / result[channel]['n_pixels']

        robust_quantile = nan_on_exception(np.quantile)

        for sums_or_means in ('sums', 'means'):
            result[channel][f'q0.5_of_cell_{sums_or_means}'] = robust_quantile(properties[f'{sums_or_means}'], 0.5)
            result[channel][f'mad_of_cell_{sums_or_means}'] = robust_quantile(
                np.abs(properties[f'{sums_or_means}'] - result[channel][f'q0.5_of_cell_{sums_or_means}']), 0.5)
            result[channel][f'q0.3_of_cell_{sums_or_means}'] = robust_quantile(properties[f'{sums_or_means}'], 0.3)
            result[channel][f'q0.7_of_cell_{sums_or_means}'] = robust_quantile(properties[f'{sums_or_means}'], 0.7)
            result[channel][f'q0.1_of_cell_{sums_or_means}'] = robust_quantile(properties[f'{sums_or_means}'], 0.1)
            result[channel][f'q0.9_of_cell_{sums_or_means}'] = robust_quantile(properties[f'{sums_or_means}'], 0.9)
            result[channel][f'mean_of_cell_{sums_or_means}'] = properties[f'{sums_or_means}'].mean()

    return result


def compute_ratios(not_infected_properties, infected_properties, channel_name_dict):
    # input should be the return value of eval_cells
    # channel_name_dict is a map from names in the table to channel keys, e.g. {'serum': 'serum_IgA'}
    not_infected_global_properties = compute_global_statistics(not_infected_properties)
    infected_global_properties = compute_global_statistics(infected_properties)
    result = dict()

    @nan_on_exception
    def serum_ratio(key, key2, serum_key):
        return (infected_global_properties[serum_key][key2]) / (not_infected_global_properties[serum_key][key])

    @nan_on_exception
    def diff_over_sum(key, key2, serum_key):
        inf, not_inf = infected_global_properties[serum_key][key], not_infected_global_properties[serum_key][key2]
        return (inf - not_inf) / (inf + not_inf)

    @nan_on_exception
    def diff(key, key2, serum_key):
        inf, not_inf = infected_global_properties[serum_key][key], not_infected_global_properties[serum_key][key2]
        return inf - not_inf

    @nan_on_exception
    def robust_z_score(sums_or_means, serum_key):
        inf = infected_global_properties[serum_key][f'q0.5_of_cell_{sums_or_means}']
        not_inf = not_infected_global_properties[serum_key][f'q0.5_of_cell_{sums_or_means}']
        mad = not_infected_global_properties[serum_key][f'mad_of_cell_{sums_or_means}']
        return (inf - not_inf) / mad

    @nan_on_exception
    def p_value(key, key2, serum_key):
        return ttest_ind(not_infected_properties[serum_key][key],
                         infected_properties[serum_key][key2],
                         equal_var=False)[1]

    # For now, I removed 'means_over_pixels'.
    # If we want to look at this, we should also consider the same with median / quantiles
    for table_key, channel_key in channel_name_dict.items():
        for sums_or_means in ('sums', 'means'):
            for key_result, key1, key2 in [
                [f'mean_of_{sums_or_means}', f'mean_of_cell_{sums_or_means}', f'mean_of_cell_{sums_or_means}'],
                [f'q0.5_of_{sums_or_means}', f'q0.5_of_cell_{sums_or_means}', f'q0.5_of_cell_{sums_or_means}'],
                [f'q0.7_vs_q0.3_of_{sums_or_means}', f'q0.7_of_cell_{sums_or_means}', f'q0.3_of_cell_{sums_or_means}'],
                [f'q0.3_vs_q0.7_of_{sums_or_means}', f'q0.3_of_cell_{sums_or_means}', f'q0.7_of_cell_{sums_or_means}'],
                [f'p-value_{sums_or_means}', sums_or_means, sums_or_means],
            ]:
                result[f'{table_key}_ratio_of_{key_result}'] = serum_ratio(key1, key2, channel_key)
                result[f'{table_key}_dos_of_{key_result}'] = diff_over_sum(key1, key2, channel_key)
                result[f'{table_key}_diff_of_{key_result}'] = diff(key1, key2, channel_key)

            result[f'{table_key}_robust_z_score_{sums_or_means}'] = robust_z_score(sums_or_means, channel_key)

        # add infected / non-infected global statistics
        for key, value in infected_global_properties[channel_key].items():
            result[f'{table_key}_infected_{key}'] = value
        for key, value in not_infected_global_properties[channel_key].items():
            result[f'{table_key}_control_{key}'] = value

    # should be included above
    # extra infected / control stuff
    #     result[f'{table_key}_infected_median'] = infected_global_properties[channel_key]['q0.5_of_cell_means']
    #     result[f'{table_key}_control_median'] = not_infected_global_properties[channel_key]['q0.5_of_cell_means']
    return result


def load_cell_outlier_dict(input_file, table_name, class_name):
    with open_file(input_file, 'r') as f:
        if not has_table(f, table_name):
            logger.warning(f"{class_name}: load_cell_outliers: did not find a cell outlier table")
            return {}
        keys, table = read_table(f, table_name)

    label_id = keys.index('label_id')
    outlier_id = keys.index('is_outlier')
    outlier_type_id = keys.index('outlier_type')
    outlier_dict = {table[ii, label_id]: (table[ii, outlier_id], table[ii, outlier_type_id])
                    for ii in range(len(table))}
    return outlier_dict


class DenoiseChannel(BatchJobOnContainer):
    def __init__(self, key_to_denoise, output_key=None, **super_kwargs):
        super().__init__(
            input_key=key_to_denoise,
            output_key=output_key if output_key is not None else key_to_denoise + '_denoised',
            **super_kwargs
        )

    def denoise(self, img):
        raise NotImplementedError

    def run(self, input_files, output_files, n_jobs=1):
        def process_image(files):
            input_file, output_file = files
            with open_file(input_file, 'r') as f:
                img = self.read_image(f, self.input_key)
            img = self.denoise(img)
            with open_file(output_file, 'a') as f:
                self.write_image(f, self.output_key, img)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(process_image, zip(input_files, output_files)),
                      desc=f'denoising {self.input_key} -> {self.output_key}',
                      total=len(output_files)))


class DenoiseByGrayscaleOpening(DenoiseChannel):
    def __init__(self, radius=5, **super_kwargs):
        super(DenoiseByGrayscaleOpening, self).__init__(**super_kwargs)
        self.structuring_element = skimage.morphology.disk(radius)

    def denoise(self, img):
        return skimage.morphology.opening(img, selem=self.structuring_element)


class DenoiseByWhiteTophat(DenoiseByGrayscaleOpening):
    def denoise(self, img):
        return skimage.morphology.white_tophat(img, selem=self.structuring_element)


def _load_image_outliers(name, table_out_path, image_outlier_table, input_files):
    if image_outlier_table is None:
        logger.warning(f"{name}: load_image_outliers: "
                       f"No image_outlier_table specified")
    if not os.path.isfile(table_out_path):
        logger.warning(f"{name}: load_image_outliers: "
                       f"did not find the hdf5 file to load an outlier table from")
        return {}
    with open_file(table_out_path, 'r') as f:
        if not has_table(f, image_outlier_table):
            logger.warning(f"{name}: load_image_outliers: did not find an image outlier table")
            return {}
        keys, table = read_table(f, image_outlier_table)

    im_name_id = keys.index('image_name')
    outlier_id = keys.index('is_outlier')
    outlier_type_id = keys.index('outlier_type')

    image_names = set(table[:, im_name_id])
    expected_names = set(in_file_to_image_name(in_file) for in_file in input_files)

    if image_names != expected_names:
        msg = f"{name}: load_image_outliers: image names from table and expected image names do not agree"
        logger.warning(msg)

    outlier_dict = {table[ii, im_name_id]: (table[ii, outlier_id], table[ii, outlier_type_id])
                    for ii in range(len(table))}
    return outlier_dict


def _get_bg_correction_dict(table_path, key_in_table, column_name, in_files):
    # if key in table is float or int, return constant background
    if isinstance(key_in_table, (int, float)):
        column_names, table = ['plate_name', column_name], np.array([['dummy_plate', key_in_table]])
    else:
        with open_file(table_path, 'r') as f:
            if not has_table(f, key_in_table):
                msg = f"Could not find table {key_in_table} in {table_path}"
                raise RuntimeError(msg)
            column_names, table = read_table(f, key_in_table)

    column_names, table = to_image_table((column_names, table), list(map(in_file_to_image_name, in_files)))
    if column_name not in column_names:
        msg = f'Did not find column {column_name} in background table columns {column_names} for {table_path}'
        raise RuntimeError(msg)

    image_names = table[:, column_names.index('image_name')]
    bg_values = table[:, column_names.index(column_name)]
    try:
        bg_values = bg_values.astype(np.float32)
    except ValueError:
        pass

    return dict(zip(image_names, bg_values))


class FindInfectedCells(BatchJobOnContainer):
    """This job finds the infected and control cells to later compute the measures on"""

    @staticmethod
    def get_table_out_key(cell_seg_key, identifier, marker_key):
        feature_table_out_key = cell_seg_key + ('' if identifier is None
                                                else f'_{identifier}') + '/' + marker_key
        return 'cell_classification/' + feature_table_out_key

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 bg_cell_seg_key=None,
                 marker_key='marker',
                 infected_threshold=6.2,
                 split_statistic='top50',
                 scale_with_mad=True,
                 bg_correction_key='plate/backgrounds',  # can also be float
                 feature_identifier=None,
                 identifier=None,
                 link_out_table=None,
                 **super_kwargs):
        self.marker_key = marker_key
        self.cell_seg_key = cell_seg_key
        self.bg_cell_seg_key = bg_cell_seg_key if bg_cell_seg_key is not None else cell_seg_key
        self.feature_table_key = cell_seg_key + ('' if feature_identifier is None
                                                 else f'_{feature_identifier}') + '/' + marker_key
        self.bg_feature_table_key = self.bg_cell_seg_key + '/' + marker_key

        self.infected_threshold = infected_threshold
        self.scale_with_mad = scale_with_mad
        self.split_statistic = split_statistic

        self.bg_correction_key = bg_correction_key

        self.output_table_key = self.get_table_out_key(cell_seg_key, identifier, marker_key)
        # we might have to link the output table to a different name
        self.link_out_table = None if link_out_table is None else self.get_table_out_key(link_out_table,
                                                                                         identifier,
                                                                                         marker_key)
        super().__init__(input_key=self.feature_table_key,
                         input_format='table',
                         output_key=self.output_table_key,
                         output_format='table',
                         identifier=identifier,
                         **super_kwargs)

    def load_feature_dict(self, in_path, for_bg=False):
        table_key = self.feature_table_key if not for_bg else self.bg_feature_table_key
        with open_file(in_path, 'r') as f:
            keys, table = self.read_table(f, table_key)
            feature_dict = {key: values for key, values in zip(keys, table.T)}
        return feature_dict

    @staticmethod
    def folder_to_table_path(folder):
        # NOTE, we call this .hdf5 to avoid pattern matching, it's a bit hacky ...
        table_file_name = os.path.split(folder)[1] + '_table.hdf5'
        return os.path.join(folder, table_file_name)

    @property
    def table_out_path(self):
        return self.folder_to_table_path(self.folder)

    def get_bg_correction_dict(self, in_files, column_name=None):
        return _get_bg_correction_dict(self.table_out_path, self.bg_correction_key, column_name, in_files)

    def get_infected_indicator(self, feature_dict, offset=None, scale=None):
        offset = 0 if offset is None else offset
        scale = 1 if scale is None else scale
        infected_indicator = feature_dict[self.split_statistic] > scale * self.infected_threshold + offset
        try:
            bg_ind = feature_dict['label_id'].tolist().index(0)
            infected_indicator[bg_ind] = False  # the background should never be classified as infected
        except ValueError:
            pass  # no bg segment
        return infected_indicator

    def get_infected_and_control_indicators(self, feature_dict, offset=None, scale=None):
        infected_indicator = self.get_infected_indicator(feature_dict, offset, scale)
        # per default, everything that is not infected is control
        control_indicator = np.logical_not(infected_indicator)
        try:
            bg_ind = feature_dict['label_id'].tolist().index(0)
            control_indicator[bg_ind] = False  # the background should never be classified as control
        except ValueError:
            pass  # no bg segment
        return infected_indicator, control_indicator

    def compute_and_save_infected_and_control(self, in_file, out_file, offset=None, scale=None):
        feature_dict = self.load_feature_dict(in_file)
        infected_indicator, control_indicator = self.get_infected_and_control_indicators(feature_dict, offset, scale)
        column_names = ['label_id', 'is_infected', 'is_control']
        table = [feature_dict['label_id'], infected_indicator, control_indicator]
        table = np.asarray(table, dtype=float).T
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_table_key, column_names, table)

    def link_result_table(self, output_file):
        with open_file(output_file, 'a') as f:
            in_key = 'tables/' + self.output_table_key
            out_key = 'tables/' + self.link_out_table
            # make a hard-link
            if out_key not in f:
                f[out_key] = f[in_key]

    def run(self, input_files, output_files, enable_tqdm=True):
        offset_dict = self.get_bg_correction_dict(input_files, column_name=f'{self.marker_key}_median')
        if self.scale_with_mad:
            scale_dict = self.get_bg_correction_dict(input_files, column_name=f'{self.marker_key}_mad')
        else:
            scale_dict = {}
        for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                            desc='finding infected cells', disable=not enable_tqdm):
            self.compute_and_save_infected_and_control(input_file, output_file,
                                                       offset=offset_dict[in_file_to_image_name(input_file)],
                                                       scale=scale_dict.get(in_file_to_image_name(input_file)))
            if self.link_out_table is not None:
                self.link_result_table(output_file)


class CellLevelAnalysisBase(BatchJobOnContainer):
    """ Base class for cell level analysis, providing access
    to the result_dict loaded from tables computed by InstanceFeatureExtraction.
    """
    def __init__(self,
                 cell_seg_key, serum_key=None, marker_key=None,
                 serum_bg_key=None, marker_bg_key=None,
                 image_input_keys=None, output_key=None,
                 validate_cell_classification=True,
                 feature_identifier=None,
                 **super_kwargs):

        self.cell_seg_key = cell_seg_key
        self.serum_bg_key = serum_bg_key
        self.marker_bg_key = marker_bg_key

        root_key = cell_seg_key if feature_identifier is None else cell_seg_key + f'_{feature_identifier}'
        input_key = [] if image_input_keys is None else list(image_input_keys)
        input_format = [] if image_input_keys is None else ['image'] * len(image_input_keys)

        if serum_key is None:
            self.serum_key = None
        else:
            self.serum_key = root_key + '/' + serum_key
            input_key.append(self.serum_key)
            input_format.append('table')

        if marker_key is None:
            self.marker_key = None
        else:
            self.marker_key = root_key + '/' + marker_key
            input_key.append(self.marker_key)
            input_format.append('table')

        if validate_cell_classification:
            self.classification_key = f'cell_classification/{cell_seg_key}/{marker_key}'
            assert self.marker_key is not None
            input_key.append(self.classification_key)
            input_format.append('table')
        else:
            self.classification_key = None

        super().__init__(input_key=input_key, input_format=input_format,
                         output_key=output_key, **super_kwargs)

    @staticmethod
    def folder_to_table_path(folder):
        # NOTE, we call this .hdf5 to avoid pattern matching, it's a bit hacky ...
        table_file_name = os.path.split(folder)[1] + '_table.hdf5'
        return os.path.join(folder, table_file_name)

    @property
    def table_out_path(self):
        return self.folder_to_table_path(self.folder)

    def load_per_cell_statistics(self, in_path, subtract_background=True, split_infected_and_control=True):
        # if multiple paths are given, concatenate the individual statistics
        if isinstance(in_path, (list, tuple)):
            in_paths = in_path
            results = [self.load_per_cell_statistics(in_path, subtract_background, split_infected_and_control)
                       for in_path in in_paths]
            if not split_infected_and_control:
                return join_cell_properties(*results)
            else:
                return (join_cell_properties(*[result[0] for result in results]),  # join infected
                        join_cell_properties(*[result[1] for result in results]))  # join control

        # only a single path is given
        with open_file(in_path, 'r') as f:
            serum_keys, serum_table = self.read_table(f, self.serum_key)
            serum_dict = {key: values for key, values in zip(serum_keys, serum_table.T)}
            marker_keys, marker_table = self.read_table(f, self.marker_key)
            marker_dict = {key: values for key, values in zip(marker_keys, marker_table.T)}
        assert np.all(serum_dict['label_id'] == marker_dict['label_id'])
        per_cell_statistics = {
            'labels': serum_dict['label_id'],
            self.serum_key: serum_dict,
            self.marker_key: marker_dict
        }
        if subtract_background:
            per_cell_statistics = self.subtract_background(per_cell_statistics, in_path)

        if not split_infected_and_control:
            return per_cell_statistics

        infected_indicator, control_indicator = self.load_infected_and_control_indicators(
            in_path, labels=per_cell_statistics['labels'])
        infected_cell_statistics = index_cell_properties(per_cell_statistics, infected_indicator)
        control_cell_statistics = index_cell_properties(per_cell_statistics, control_indicator)

        return infected_cell_statistics, control_cell_statistics

    def load_infected_and_control_indicators(self, in_path, labels):
        with open_file(in_path, 'r') as f:
            column_names, table = self.read_table(f, self.classification_key)
        infected_dict = get_column_dict(column_names, table, 'is_infected')
        control_dict = get_column_dict(column_names, table, 'is_control')
        if hasattr(self, 'load_cell_outliers'):
            # remove cell outliers from infected / control and thereby the analysis
            cell_outlier_dict = self.load_cell_outliers(in_path)
            for label, (is_outlier, _) in cell_outlier_dict.items():
                if is_outlier == 1:
                    infected_dict[label] = 0
                    control_dict[label] = 0
        # convert dicts to lists
        infected_indicator = np.array([infected_dict.get(label, 0) for label in labels])
        control_indicator = np.array([control_dict.get(label, 0) for label in labels])
        return infected_indicator, control_indicator

    @property
    def bg_dict(self):
        if not hasattr(self, '_bg_dict'):
            # TODO: this is not ideal.. get the list of input files differently
            input_folder = self.input_folder
            in_pattern = os.path.join(input_folder, self.input_pattern)
            input_files = glob(in_pattern)

            def channel_to_bg_column(channel):
                # because e.g. serum_key = 'cell_segmentation/serum_IgA'
                return f'{channel.split("/")[-1]}_median'
            self._bg_dict = {channel: _get_bg_correction_dict(self.table_out_path,
                                                              bg_key,
                                                              channel_to_bg_column(channel),
                                                              input_files)
                             for channel, bg_key in ((self.serum_key, self.serum_bg_key),
                                                     (self.marker_key, self.marker_bg_key))}
        return self._bg_dict

    def subtract_background(self, per_cell_statistics, in_path):
        per_cell_statistics = deepcopy(per_cell_statistics)
        image_name = in_file_to_image_name(in_path)
        for channel, bg_key in ((self.serum_key, self.serum_bg_key), (self.marker_key, self.marker_bg_key)):
            if bg_key is None:
                continue
            bg_offset = self.bg_dict[channel][image_name]
            channel_statistics = per_cell_statistics[channel]
            for key in channel_statistics.keys():
                if key in ['means', 'medians'] or key.startswith('top') or key.startswith('quantile'):
                    channel_statistics[key] -= bg_offset
                elif key == 'sums':
                    # sums are special case
                    channel_statistics['sums'] -= bg_offset * channel_statistics['sizes']
                elif key in ['label_id', 'mads', 'sizes'] or '_bg_' in key:
                    pass
                else:
                    assert False, f"No background subtraction rule specified for key '{key}'"
        return per_cell_statistics

    def get_stat_dict(self, infected_cell_statistics, control_cell_statistics):
        stat_dict = compute_ratios(control_cell_statistics, infected_cell_statistics,
                                   channel_name_dict=dict(serum=self.serum_key, marker=self.marker_key))
        if hasattr(self, 'score_name') and self.score_name is not None:
            stat_dict['score'] = stat_dict[self.score_name]
        stat_dict['n_infected'] = len(next(iter(infected_cell_statistics[self.serum_key].values())))
        stat_dict['n_control'] = len(next(iter(control_cell_statistics[self.serum_key].values())))

        # this only accounts for cells that were either classified as infected or control
        stat_dict['n_cells'] = stat_dict['n_infected'] + stat_dict['n_control']
        stat_dict['fraction_infected'] = nan_on_exception(lambda: stat_dict['n_infected'] / stat_dict['n_cells'])()

        stat_dict['cell_size_median_infected'] = np.median(infected_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_mean_infected'] = np.mean(infected_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_median_control'] = np.median(control_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_mean_control'] = np.mean(control_cell_statistics[self.serum_key]['sizes'])

        return stat_dict

    def group_images_by_well(self, input_files):
        images_per_well = defaultdict(list)
        for in_file in input_files:
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            well_name = image_name_to_well_name(image_name)
            images_per_well[well_name].append(in_file)
        return images_per_well


class CellLevelAnalysisWithTableBase(CellLevelAnalysisBase):
    """
    """
    def __init__(self, table_out_keys, check_image_outputs=False, **super_kwargs):
        self.table_out_keys = table_out_keys
        self.check_image_outputs = check_image_outputs
        super().__init__(**super_kwargs)

    def check_table(self, log_on_fail):
        cls_name = 'CellLevelAnalysisWithTableBase'
        table_path = self.table_out_path
        if not os.path.exists(table_path):
            log_on_fail(f'{cls_name}: check failed: could not find {table_path}')
            return False
        with open_file(table_path, 'r') as f:
            if not all(has_table(f, key) for key in self.table_out_keys):
                msg = f'{cls_name}: check failed: could not find all expected tables {self.table_out_keys}'
                log_on_fail(msg)
                return False
        return True

    # we only write a single output file, so need to over-write the output validation and output checks
    def check_output(self, path, log_on_fail=logger.debug):
        have_table = self.check_table(log_on_fail)
        if self.check_image_outputs:
            return have_table and super().check_output(path, log_on_fail)
        else:
            return have_table

    def validate_outputs(self, output_files, folder, status, ignore_failed_outputs):
        have_table = self.check_table(logger.warning)
        if self.check_image_outputs:
            return have_table and super().validate_outputs(output_files,
                                                           folder, status,
                                                           ignore_failed_outputs)
        else:
            return have_table


class CellLevelAnalysis(CellLevelAnalysisWithTableBase):
    """
    """
    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate/backgrounds',
                 marker_bg_key='plate/backgrounds',
                 score_name='serum_ratio_of_q0.5_of_means',
                 write_summary_images=False,
                 infected_cell_mask_key='infected_cell_mask',
                 serum_per_cell_mean_key='serum_per_cell_mean',
                 edge_key='cell_segmentation_edges',
                 outlier_cell_mask_key='outlier_cell_mask',
                 cell_outlier_table_name='outliers',
                 image_outlier_table='images/outliers',
                 well_outlier_table='wells/outliers',
                 identifier=None,
                 **super_kwargs):

        # table keys in the plate-wise *_table.hdf5
        self.image_table_key = f'images/{identifier if identifier is not None else "default"}'
        self.well_table_key = f'wells/{identifier if identifier is not None else "default"}'

        self.score_name = score_name
        self.write_summary_images = write_summary_images

        # FIXME this does not take the identifier into account
        self.image_outlier_table = image_outlier_table
        self.well_outlier_table = well_outlier_table

        if self.write_summary_images:
            output_key = [infected_cell_mask_key,
                          serum_per_cell_mean_key,
                          edge_key,
                          outlier_cell_mask_key]
            self.edge_key = edge_key
            self.infected_cell_mask_key = infected_cell_mask_key
            self.serum_per_cell_mean_key = serum_per_cell_mean_key
            self.outlier_cell_mask_key = outlier_cell_mask_key
        else:
            output_key = None

        super().__init__(table_out_keys=[self.image_table_key,
                                         self.well_table_key],
                         check_image_outputs=self.write_summary_images,
                         cell_seg_key=cell_seg_key,
                         serum_key=serum_key,
                         marker_key=marker_key,
                         serum_bg_key=serum_bg_key,
                         marker_bg_key=marker_bg_key,
                         output_key=output_key,
                         identifier=identifier,
                         **super_kwargs)

        output_group = cell_seg_key if self.identifier is None else cell_seg_key + '_' + self.identifier
        self.cell_outlier_table = output_group + '/' + serum_key + '_' + cell_outlier_table_name

    def load_image_outliers(self, input_files):
        return _load_image_outliers(self.name, self.table_out_path, self.image_outlier_table, input_files)

    def load_well_outliers(self, well_names):
        with open_file(self.table_out_path, 'r') as f:
            if not self.has_table(f, self.well_outlier_table):
                logger.warning(f"{self.name}: load_well_outliers: did not find an well outlier table")
                return {}
            keys, table = self.read_table(f, self.well_outlier_table)

        well_name_id = keys.index('well_name')
        outlier_id = keys.index('is_outlier')
        outlier_type_id = keys.index('outlier_type')

        well_names = set(table[:, well_name_id])
        expected_names = set(well_names)

        if well_names != expected_names:
            msg = f"{self.name}: load_well_outliers: well names from table and expected well names do not agree"
            logger.warning(msg)

        outlier_dict = {table[ii, well_name_id]: (table[ii, outlier_id], table[ii, outlier_type_id])
                        for ii in range(len(table))}
        return outlier_dict

    def load_cell_outliers(self, input_file):
        return load_cell_outlier_dict(input_file, self.cell_outlier_table, self.name)

    def write_image_table(self, input_files):

        image_outlier_dict = self.load_image_outliers(input_files)

        column_names = ['image_name', 'site_name', 'is_outlier', 'outlier_type', 'n_outlier_cells']
        table = []

        for ii, in_file in enumerate(tqdm(input_files, desc='generating image table')):
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]

            infected_cell_statistics, control_cell_statistics = self.load_per_cell_statistics(in_file)

            # get all the statistics for this image and their names
            stat_dict = self.get_stat_dict(infected_cell_statistics, control_cell_statistics)

            # Set the main score to nan if this image is an outlier
            outlier, outlier_type = image_outlier_dict.get(image_name, (-1, 'not_checked'))
            stat_dict['score'] = np.nan if (outlier == 1 or stat_dict['score'] is None) else stat_dict['score']

            stat_names, stat_list = map(list, zip(*stat_dict.items()))
            if ii == 0:
                column_names += stat_names

            site_name = image_name_to_site_name(image_name)

            # get number of ignored outlier cells
            n_outlier_cells = sum(1 for v in self.load_cell_outliers(in_file).values() if v[0] == 1)

            table.append([image_name, site_name, outlier, outlier_type, n_outlier_cells] + stat_list)

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        # set image name to non-visible for the plateViewer (something else?)
        visible = np.ones(n_cols, dtype='uint8')
        visible[0] = False

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.image_table_key, column_names, table, visible, force_write=True)

        return table, column_names

    def write_well_table(self, input_files):

        # group input files per well
        input_files_per_well = self.group_images_by_well(input_files)
        well_names = list(input_files_per_well.keys())
        well_names.sort()
        n_wells = len(well_names)

        well_outlier_dict = self.load_well_outliers(well_names)
        image_outlier_dict = self.load_image_outliers(input_files)

        column_names = None
        initial_column_names = ['well_name', 'n_outlier_images', 'n_outlier_cells', 'is_outlier', 'outlier_type']

        table = []
        empty_wells = []

        for well_name, in_files_for_current_well in tqdm(input_files_per_well.items(), desc='generating well table'):
            n_total = len(in_files_for_current_well)
            image_names_for_current_well = [os.path.splitext(os.path.split(in_file)[1])[0]
                                            for in_file in in_files_for_current_well]

            in_files_for_current_well = [in_file for in_file, im_name in zip(in_files_for_current_well,
                                                                             image_names_for_current_well)
                                         if not image_outlier_dict.get(im_name, (-1, ''))[0] == 1]

            if len(in_files_for_current_well) == 0:
                logger.info(f'Well {well_name} consists entirely of outlier images and will be marked as outlier well.')
                # if we have no input images, we cannot run 'load_per_cell_statistics'
                # -> need to skip computation for this well
                # we also might not have the correct 'column_names' yet, because we may only got empty wells so far.
                # -> we just append the well name, mark this well as empty and fix the issue later
                table.append([well_name])
                empty_wells.append(well_name)
                continue

            n_outlier_images = n_total - len(in_files_for_current_well)

            well_is_outlier, outlier_type = well_outlier_dict.get(well_name, (-1, 'not checked'))
            if well_is_outlier == 1:
                logger.info(f'Well {well_name} was flagged as outlier.')

            infected_cell_statistics, control_cell_statistics = self.load_per_cell_statistics(in_files_for_current_well)

            # get all the statistics for this well and their names
            stat_dict = self.get_stat_dict(infected_cell_statistics, control_cell_statistics)

            # get the main score, which is the measure computed for `score_name`, but set to
            # nan if this image is an outlier
            stat_dict['score'] = np.nan if (stat_dict['score'] is None or well_is_outlier == 1) else stat_dict['score']

            stat_names, stat_list = map(list, zip(*stat_dict.items()))
            if column_names is None:
                column_names = initial_column_names + list(stat_names)

            # get number of ignored outlier cells
            n_outlier_cells = sum(sum(1 for v in self.load_cell_outliers(in_file).values() if v[0] == 1)
                                  for in_file in in_files_for_current_well)
            table.append([well_name, n_outlier_images, n_outlier_cells, well_is_outlier, outlier_type] + stat_list)

        # check if we have to insert to the table for empty wells
        if len(empty_wells) > 0:
            if len(empty_wells) == n_wells:
                logger.warning(f"{self.name}: all wells are empty (all images are outliers)")
                # make dummy table
                table = [[well_name, np.nan, np.nan, 1, 'all images are outliers'] for well_name in well_names]
            else:
                n_cols = len(column_names)
                if not all(len(row) in (1, n_cols) for row in table):
                    raise RuntimeError(f"Invalid number of columns in well table, expected 1 or {n_cols}")
                outlier_row = [np.nan, np.nan, 1, 'all images are outliers']
                outlier_row += [np.nan] * (n_cols - len(outlier_row) - 1)
                assert len(outlier_row) == n_cols - 1
                table = [row if len(row) == n_cols else row + outlier_row for row in table]

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.well_table_key, column_names, table, force_write=True)

        return table, column_names

    # write the image and well score attributes
    def write_image_and_well_information(self, files,
                                         image_table, image_columns,
                                         well_table, well_columns):

        image_score_index = image_columns.index('score')
        image_information = dict(zip(image_table[:, 0], image_table[:, image_score_index]))

        well_score_index = well_columns.index('score')
        well_information = dict(zip(well_table[:, 0], well_table[:, well_score_index]))

        for path in files:
            name = os.path.splitext(os.path.split(path)[1])[0]
            well_name = image_name_to_well_name(name)
            write_image_information(path,
                                    image_information=str(image_information[name]),
                                    well_information=str(well_information.get(well_name, np.nan)))

    def write_summary_image(self, in_path, out_path):

        with open_file(in_path, 'r') as f:
            cell_seg = self.read_image(f, self.cell_seg_key)

        label_ids = self.load_per_cell_statistics(in_path, False, False)['labels']

        # make the segmentation edge image
        seg_edges = seg_to_edges(cell_seg).astype('uint8')

        # make a label mask for the cells detected as outliers
        cell_outlier_dict = self.load_cell_outliers(in_path)
        outlier_indicator = np.array([cell_outlier_dict.get(lid, (0, ""))[0] for lid in label_ids], dtype='int8')
        outlier_indicator[outlier_indicator == -1] = 0
        outlier_indicator[0] = 0
        assert len(outlier_indicator) == len(label_ids)
        outlier_label_ids = label_ids[outlier_indicator.astype('bool')]
        outlier_mask = np.isin(cell_seg, outlier_label_ids).astype(cell_seg.dtype)
        # mark the seg edges in a different color
        outlier_mask[seg_edges == 1] = 2

        # make a label mask for the infected cells
        infected_indicator, _ = self.load_infected_and_control_indicators(in_path, label_ids)
        assert len(label_ids) == len(infected_indicator), f'{len(label_ids)} != {len(infected_indicator)}'
        infected_label_ids = label_ids[infected_indicator.astype('bool')]  # cast to bool again to be sure

        infected_mask = np.isin(cell_seg, infected_label_ids).astype(cell_seg.dtype)
        # mark the outliers in a different color
        infected_mask[outlier_mask == 1] = 3
        # mark the seg edges in a different color
        infected_mask[seg_edges == 1] = 2

        # meak an image with the mean serum intensity
        result = self.load_per_cell_statistics(in_path, subtract_background=True,
                                               split_infected_and_control=False)
        mean_serum_image = np.zeros_like(cell_seg, dtype=np.float32)
        for label, intensity in zip(filter(lambda x: x != 0, label_ids),
                                    result[self.serum_key]['means']):
            mean_serum_image[cell_seg == label] = intensity

        with open_file(out_path, 'a') as f:
            # we need to use nearest down-sampling for the mean serum images,
            # because while these are float values, they should not be interpolated
            self.write_image(f, self.serum_per_cell_mean_key, mean_serum_image,
                             settings={'use_nearest': True})
            self.write_image(f, self.infected_cell_mask_key, infected_mask)
            self.write_image(f, self.outlier_cell_mask_key, outlier_mask)
            self.write_image(f, self.edge_key, seg_edges)

    def run(self, input_files, output_files, n_jobs=1):
        image_table, image_columns = self.write_image_table(input_files)
        well_table, well_columns = self.write_well_table(input_files)
        self.write_image_and_well_information(output_files, image_table, image_columns,
                                              well_table, well_columns)

        if self.write_summary_images:
            with futures.ThreadPoolExecutor(n_jobs) as tp:
                list(tqdm(tp.map(self.write_summary_image, input_files, output_files),
                          desc='write cell level analysis summary images',
                          total=len(input_files)))
