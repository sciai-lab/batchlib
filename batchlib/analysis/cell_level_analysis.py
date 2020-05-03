import os
from copy import copy, deepcopy
from functools import partial
from collections import defaultdict

import numpy as np
import skimage.morphology
import torch
from tqdm.auto import tqdm

from batchlib.util.logging import get_logger
from ..base import BatchJobOnContainer
from ..util.image import seg_to_edges
from ..util.io import open_file, image_name_to_site_name, image_name_to_well_name, write_image_information

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


def compute_global_statistics(cell_properties):
    result = dict()
    for channel, properties in cell_properties.items():
        if not isinstance(properties, dict):
            continue
        result[channel] = dict()
        result[channel]['n_pixels'] = properties['sizes'].sum()
        result[channel]['sum'] = properties['sums'].sum()
        result[channel]['global_mean'] = result[channel]['sum'] / result[channel]['n_pixels']

        def robust_quantile(arr, q):
            try:
                result = np.quantile(arr, q)
            except Exception:
                result = np.nan
            return result

        result[channel]['q0.5_of_cell_sums'] = robust_quantile(properties['sums'], 0.5)
        result[channel]['mad_of_cell_sums'] = robust_quantile(np.abs(properties['sums'] -
                                                                     result[channel]['q0.5_of_cell_sums']), 0.5)
        result[channel]['q0.5_of_cell_means'] = robust_quantile(properties['means'], 0.5)
        result[channel]['mad_of_cell_means'] = robust_quantile(np.abs(properties['means'] -
                                                                      result[channel]['q0.5_of_cell_means']), 0.5)
        result[channel]['q0.3_of_cell_means'] = robust_quantile(properties['means'], 0.3)
        result[channel]['q0.7_of_cell_means'] = robust_quantile(properties['means'], 0.7)
        result[channel]['q0.1_of_cell_means'] = robust_quantile(properties['means'], 0.1)
        result[channel]['q0.9_of_cell_means'] = robust_quantile(properties['means'], 0.9)
        result[channel]['cell_mean'] = properties['means'].mean()
        result[channel]['cell_sum'] = properties['sums'].mean()

    return result


def compute_ratios(control_properties, infected_properties, serum_key='serum'):
    # input should be the return value of eval_cells
    control_global_properties = compute_global_statistics(control_properties)
    infected_global_properties = compute_global_statistics(infected_properties)
    result = dict()

    def serum_ratio(key, key2=None):
        key2 = key if key2 is None else key
        try:
            result = (infected_global_properties[serum_key][key2]) / (control_global_properties[serum_key][key])
        except Exception:
            result = np.nan
        return result

    def diff_over_sum(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties[serum_key][key], control_global_properties[serum_key][key2]
            result = (inf - not_inf) / (inf + not_inf)
        except Exception:
            result = np.nan
        return result

    def diff(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties[serum_key][key], control_global_properties[serum_key][key2]
            result = inf - not_inf
        except Exception:
            result = np.nan
        return result

    def robust_z_score(mode='sums'):
        assert mode in ('sums', 'means')
        try:
            inf = infected_global_properties[serum_key][f'q0.5_of_cell_{mode}']
            not_inf = control_global_properties[serum_key][f'q0.5_of_cell_{mode}']
            mad = control_global_properties[serum_key][f'mad_of_cell_{mode}']
            result = (inf - not_inf) / mad
        except Exception:
            result = np.nan
        return result

    for key_result, key1, key2 in [
        ['global_means', 'global_mean', 'global_mean'],
        ['mean_of_means', 'cell_mean', 'cell_mean'],
        ['mean_of_sums', 'cell_sum', 'cell_sum'],
        ['median_of_means', 'q0.5_of_cell_means', 'q0.5_of_cell_means'],
        ['median_of_sums', 'q0.5_of_cell_sums', 'q0.5_of_cell_sums'],
        ['q0.7_vs_q0.3', 'q0.7_of_cell_means', 'q0.3_of_cell_means'],
        ['q0.3_vs_q0.7', 'q0.3_of_cell_means', 'q0.7_of_cell_means'],
    ]:
        result[f'ratio_of_{key_result}'] = serum_ratio(key1, key2)
        result[f'dos_of_{key_result}'] = diff_over_sum(key1, key2)
        result[f'diff_of_{key_result}'] = diff(key1, key2)

    # add infected / non-infected global statistics
    for key, value in infected_global_properties[serum_key].items():
        result[f'infected_{key}'] = value
    for key, value in control_global_properties[serum_key].items():
        result[f'control_{key}'] = value

    for mode in ('means', 'sums'):
        result[f'robust_z_score_{mode}'] = robust_z_score(mode)

    # extra infected / control stuff
    result['infected_mean'] = infected_global_properties[serum_key]['global_mean']
    result['infected_median'] = infected_global_properties[serum_key]['q0.5_of_cell_means']
    result['control_mean'] = control_global_properties[serum_key]['global_mean']
    result['control_median'] = control_global_properties[serum_key]['q0.5_of_cell_means']
    return result


class DenoiseChannel(BatchJobOnContainer):
    def __init__(self, key_to_denoise, output_key=None, output_ext='.h5'):
        super(DenoiseChannel, self).__init__(
            output_ext=output_ext,
            input_key=key_to_denoise,
            output_key=output_key if output_key is not None else key_to_denoise + '_denoised'
        )

    def denoise(self, img):
        raise NotImplementedError

    def run(self, input_files, output_files):
        for input_file, output_file in zip(tqdm(input_files, f'denoising {self.input_key} -> {self.output_key}'),
                                           output_files):
            with open_file(input_file, 'r') as f:
                img = self.read_image(f, self.input_key)
            img = self.denoise(img)
            with open_file(output_file, 'a') as f:
                self.write_image(f, self.output_key, img)


class DenoiseByGrayscaleOpening(DenoiseChannel):
    def __init__(self, radius=5, **super_kwargs):
        super(DenoiseByGrayscaleOpening, self).__init__(**super_kwargs)
        self.structuring_element = skimage.morphology.disk(radius)

    def denoise(self, img):
        return skimage.morphology.opening(img, selem=self.structuring_element)


class InstanceFeatureExtraction(BatchJobOnContainer):
    def __init__(self,
                 channel_keys=('serum', 'marker'),
                 nuc_seg_key_to_ignore=None,
                 cell_seg_key='cell_segmentation',
                 identifier=None):

        self.channel_keys = tuple(channel_keys)
        self.nuc_seg_key_to_ignore = nuc_seg_key_to_ignore
        self.cell_seg_key = cell_seg_key

        # all inputs should be 2d
        input_ndim = [2] * (1 + len(channel_keys) + (1 if nuc_seg_key_to_ignore else 0))

        # tables are per default saved at tables/cell_segmentation/channel in the container
        output_group = cell_seg_key if identifier is None else cell_seg_key + '_' + identifier
        self.output_table_keys = [output_group + '/' + channel for channel in channel_keys]
        super().__init__(input_key=list(self.channel_keys) + [self.cell_seg_key] +
                                       ([self.nuc_seg_key_to_ignore] if self.nuc_seg_key_to_ignore is not None else []),
                         input_ndim=input_ndim,
                         output_key=['tables/' + key for key in self.output_table_keys],
                         identifier=identifier)

    def load_sample(self, path, device):
        with open_file(path, 'r') as f:
            channels = [self.read_image(f, key) for key in self.channel_keys]
            cell_seg = self.read_image(f, self.cell_seg_key)
            if self.nuc_seg_key_to_ignore is not None:
                nucleus_seg = self.read_image(f, self.nuc_seg_key_to_ignore)

        channels = [torch.FloatTensor(channel.astype(np.float32)).to(device) for channel in channels]
        cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)

        if self.nuc_seg_key_to_ignore is not None:
            nucleus_seg = torch.LongTensor(nucleus_seg.astype(np.int32)).to(device)
            cell_seg[nucleus_seg != 0] = -1  # negative labels are ignored in eval_cells

        return channels, cell_seg

    def get_per_instance_statistics(self, data, seg, labels):
        per_cell_values = [data[seg == label] for label in labels]
        sums = data.new([arr.sum() for arr in per_cell_values])
        means = data.new([arr.mean() for arr in per_cell_values])
        instance_sizes = data.new([len(arr.view(-1)) for arr in per_cell_values])
        top50 = np.array([0 if len(t) < 50 else t.topk(50)[0][-1].item()
                          for t in per_cell_values])
        top30 = np.array([0 if len(t) < 30 else t.topk(30)[0][-1].item()
                          for t in per_cell_values])
        top10 = np.array([0 if len(t) < 10 else t.topk(10)[0][-1].item()
                          for t in per_cell_values])
        medians = np.array([t.median().item()
                            for t in per_cell_values])
        mads = np.array([(t - median).abs().median().item()
                         for t, median in zip(per_cell_values, medians)])

        # when adding a new stat, make sure that it's BG is subtracted in CellLevelAnalysis.subtract_background()
        # convert to numpy here
        return dict(sums=sums.cpu().numpy(),
                    means=means.cpu().numpy(),
                    medians=medians,
                    mads=mads,
                    sizes=instance_sizes.cpu().numpy(),
                    top50=top50,
                    top30=top30,
                    top10=top10)

    def eval_cells(self, channels, cell_seg,
                   ignore_label=0):
        # all segs have shape H, W
        shape = cell_seg.shape
        for channel in list(channels):
            assert channel.shape == shape

        # include background as instance with label 0
        labels = torch.sort(torch.unique(cell_seg))[0]
        labels = labels[labels >= 0]  # ignore nuclei with label -1

        cell_properties = dict()
        for key, channel in zip(self.channel_keys, channels):
            cell_properties[key] = self.get_per_instance_statistics(channel, cell_seg, labels)
        cell_properties['labels'] = labels.cpu().numpy()

        return cell_properties

    def get_bg_segment(self, in_file, ignore_label=0, device='cpu'):
        channels, cell_seg = self.load_sample(in_file, device=device)
        return torch.stack(channels)[:, cell_seg == ignore_label]

    # this is what should be run for each h5 file
    def save_all_stats(self, in_file, out_file,
                       bg_per_image_stats, bg_per_well_stats, bg_plate_stats, device):
        sample = self.load_sample(in_file, device=device)
        per_cell_statistics = self.eval_cells(*sample)

        labels = per_cell_statistics['labels']
        for i, (channel, output_key) in enumerate(zip(self.channel_keys, self.output_table_keys)):
            columns = ['label_id']
            table = [list(labels)]

            # add background stats to all cells (in case we want a different bg for each cell at some point)
            n_cells = len(table[0])
            # per-plate bg stats
            for key, values in bg_plate_stats.items():
                columns.append(f'plate_bg_{key}')
                table.append([values[i]] * n_cells)
            # per-well bg stats
            well = image_name_to_well_name(os.path.basename(in_file))
            for key, values in bg_per_well_stats[well].items():
                columns.append(f'well_bg_{key}')
                table.append([values[i]] * n_cells)
            # per-image bg stats
            for key, values in bg_per_image_stats[in_file].items():
                columns.append(f'image_bg_{key}')
                table.append([values[i]] * n_cells)

            # actual per-cell stats
            for key, values in per_cell_statistics[channel].items():
                columns.append(key)
                table.append([v if v is not None else np.nan for v in values])

            # transpose table to have shape (n_cells, n_features)
            table = np.asarray(table, dtype=float).T
            with open_file(out_file, 'a') as f:
                self.write_table(f, output_key, columns, table)

    def get_bg_stats(self, bg_values):
        # bg_vales should have shape n_channels, n_pixels
        bg_values = bg_values.cpu().numpy().astype(np.float32)
        medians = np.median(bg_values, axis=1)
        mads = np.median(np.abs(bg_values - medians[:, None]), axis=1)

        return {'median': medians, 'mad': mads}

    def run(self, input_files, output_files, gpu_id=None):
        # first, get plate wide and per-well background statistics
        logger.info('computing background statistics')
        bg_dict = {file: self.get_bg_segment(file, device='cpu') for file in input_files}
        bg_per_image_stats = {file: self.get_bg_stats(bg_segments) for file, bg_segments in bg_dict.items()}
        bg_per_well_dict = defaultdict(list)
        for file, bg_segment in bg_dict.items():
            bg_per_well_dict[image_name_to_well_name(os.path.basename(file))].append(bg_segment)
        bg_per_well_dict = {well: torch.cat(bg_segments, dim=1) for well, bg_segments in bg_per_well_dict.items()}
        bg_per_well_stats = {well: self.get_bg_stats(bg_pixels) for well, bg_pixels in bg_per_well_dict.items()}
        bg_plate_stats = self.get_bg_stats(torch.cat(list(bg_per_well_dict.values()), dim=1))
        with torch.no_grad():
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                device = torch.device(0)
            else:
                device = torch.device('cpu')
            _save_all_stats = partial(self.save_all_stats, device=device)
            for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                                desc='extracting cell-level features'):
                _save_all_stats(input_file, output_file, bg_per_image_stats, bg_per_well_stats, bg_plate_stats)


class FindInfectedCells(BatchJobOnContainer):
    """This job finds the infected and control cells to later compute the measures on"""
    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 marker_key='marker',
                 infected_threshold=250, split_statistic='top50',
                 infected_threshold_scale_key=None,
                 bg_correction_key=None,
                 per_cell_bg_correction=True,
                 identifier=None,
                 **super_kwargs):
        self.marker_key = marker_key
        self.cell_seg_key = cell_seg_key
        self.feature_table_key = cell_seg_key + '/' + marker_key

        self.infected_threshold = infected_threshold
        self.infected_threshold_scale_key = infected_threshold_scale_key
        self.split_statistic = split_statistic

        self.bg_correction_key = bg_correction_key
        self.per_cell_bg_correction = per_cell_bg_correction

        # infected are per default saved at tables/cell_classification/cell_segmentation/marker_key in the container
        self.output_table_key = 'cell_classification/' + self.feature_table_key + \
                                ('' if identifier is None else '_' + identifier)
        super().__init__(input_key='tables/' + self.feature_table_key,
                         output_key='tables/' + self.output_table_key,
                         identifier=identifier,
                         **super_kwargs)

    def load_feature_dict(self, in_path):
        with open_file(in_path, 'r') as f:
            keys, table = self.read_table(f, self.feature_table_key)
            feature_dict = {key: values for key, values in zip(keys, table.T)}
        return feature_dict

    def get_bg_correction(self, feature_dict, bg_correction_key=None):
        bg_correction_key = self.bg_correction_key if bg_correction_key is None else bg_correction_key
        if bg_correction_key:
            if self.per_cell_bg_correction:
                offset = feature_dict[bg_correction_key]
            else:
                try:
                    bg_ind = feature_dict['label_id'].tolist().index(0)
                    offset = feature_dict[bg_correction_key][bg_ind]
                except ValueError:
                    offset = np.nan
        else:
            offset = 0
        return offset

    def get_infected_indicator(self, feature_dict):
        offset = self.get_bg_correction(feature_dict)
        if self.infected_threshold_scale_key is not None:
            scale = feature_dict[self.infected_threshold_scale_key]
        else:
            scale = 1
        infected_indicator = feature_dict[self.split_statistic] > scale * self.infected_threshold + offset

        try:
            bg_ind = feature_dict['label_id'].tolist().index(0)
            infected_indicator[bg_ind] = False  # the background should never be classified as infected
        except ValueError:
            pass  # no bg segment
        return infected_indicator

    def get_infected_and_control_indicators(self, feature_dict):
        infected_indicator = self.get_infected_indicator(feature_dict)
        # per default, everything that is not infected is control
        control_indicator = np.logical_not(infected_indicator)
        try:
            bg_ind = feature_dict['label_id'].tolist().index(0)
            control_indicator[bg_ind] = False  # the background should never be classified as control
        except ValueError:
            pass  # no bg segment
        return infected_indicator, control_indicator

    def compute_and_save_infected_and_control(self, in_file, out_file):
        feature_dict = self.load_feature_dict(in_file)
        infected_indicator, control_indicator = self.get_infected_and_control_indicators(feature_dict)
        column_names = ['label_id', 'is_infected', 'is_control']
        table = [feature_dict['label_id'], infected_indicator, control_indicator]
        table = np.asarray(table, dtype=float).T
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_table_key, column_names, table)

    def run(self, input_files, output_files):
        for input_file, output_file in tqdm(list(zip(input_files, output_files)), desc='finding infected cells'):
            self.compute_and_save_infected_and_control(input_file, output_file)


class CellLevelAnalysisBase(BatchJobOnContainer):
    """ Base class for cell level analysis, providing access
    to the result_dict loaded from tables computed by InstanceFeatureExtraction.
    """
    def __init__(self,
                 cell_seg_key, serum_key, marker_key,
                 serum_bg_key, marker_bg_key,
                 output_key=None, **super_kwargs):

        self.cell_seg_key = cell_seg_key
        self.serum_bg_key = serum_bg_key
        self.marker_bg_key = marker_bg_key

        # TODO allow for serum and marker data to come from different segmentations
        self.serum_key = cell_seg_key + '/' + serum_key
        self.marker_key = cell_seg_key + '/' + marker_key
        self.classification_key = 'cell_classification/' + cell_seg_key + '/' + marker_key

        super().__init__(input_key=[f'tables/{key}'
                                    for key in (self.serum_key, self.marker_key, self.classification_key)],
                         output_key=output_key,
                         **super_kwargs)

    # in the long run we should merge this into BatchJobOnContainer somehow
    def validate_input(self, path):
        if not os.path.exists(path):
            logger.warning(f'{self.name}: validate_input failed: {path} does not exist')
            return False

        exp_keys = self._input_exp_key
        if exp_keys is None:
            return True
        with open_file(path, 'r') as f:
            for key in exp_keys:
                if key not in f:
                    logger.warning(f'{self.name}: validate_input failed: could not find {key} in {path}')
                    return False
                g = f[key]
                if ('cells' not in g) or ('columns' not in g):
                    msg = f"{self.name}: validate_input failed: could not find 'cells' or 'columns' in {path}:{key}"
                    logger.warning(msg)
                    return False
        return True

    @staticmethod
    def folder_to_table_path(folder, identifier):
        # NOTE, we call this .hdf5 to avoid pattern matching, it's a bit hacky ...
        table_file_name = os.path.split(folder)[1] + \
                          ('_table.hdf5' if identifier is None else f'_table_{identifier}.hdf5')
        return os.path.join(folder, table_file_name)

    @property
    def table_out_path(self):
        return self.folder_to_table_path(self.folder, self.identifier)

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
            per_cell_statistics = self.subtract_background(per_cell_statistics)

        if not split_infected_and_control:
            return per_cell_statistics

        infected_indicator, control_indicator = self.load_infected_and_control_indicators(in_path)
        infected_cell_statistics = index_cell_properties(per_cell_statistics, infected_indicator)
        control_cell_statistics = index_cell_properties(per_cell_statistics, control_indicator)

        return infected_cell_statistics, control_cell_statistics

    def load_infected_and_control_indicators(self, in_path):
        with open_file(in_path, 'r') as f:
            column_names, table = self.read_table(f, self.classification_key)
        infected_indicator = table[:, column_names.index('is_infected')]
        control_indicator = table[:, column_names.index('is_control')]
        if hasattr(self, 'load_cell_outliers'):
            # remove cell outliers from infected / control and thereby the analysis
            cell_outlier_dict = self.load_cell_outliers(in_path)
            labels = table[:, column_names.index('label_id')].astype(np.int32)
            for i, label in enumerate(labels):
                if cell_outlier_dict.get(label, (-1, ""))[0] == 1:
                    infected_indicator[i] = 0
                    control_indicator[i] = 0
        return infected_indicator, control_indicator

    def subtract_background(self, per_cell_statistics):
        per_cell_statistics = deepcopy(per_cell_statistics)
        for channel, bg_key in ((self.serum_key, self.serum_bg_key), (self.marker_key, self.marker_bg_key)):
            if bg_key is None:
                continue
            channel_statistics = per_cell_statistics[channel]
            try:
                bg_offset = next(iter(channel_statistics[bg_key]))
            except StopIteration:
                # no cells, nothing to do
                continue
            for key in channel_statistics.keys():
                if key in ['means', 'medians', 'top50', 'top30', 'top10']:
                    channel_statistics[key] -= bg_offset
                elif key == 'sums':
                    # sums are special case
                    channel_statistics['sums'] -= bg_offset * channel_statistics['sizes']
                elif key in ['label_id', 'mads', 'sizes'] or '_bg_' in key:
                    pass
                else:
                    assert False, f"No background subtraction rule specified for key '{key}'"
        return per_cell_statistics

    def get_stat_dict(self, infected_cell_statistics, control_cell_statistics, bg_keys=None):
        stat_dict = compute_ratios(control_cell_statistics, infected_cell_statistics, serum_key=self.serum_key)
        stat_dict['score'] = stat_dict[self.score_name]
        stat_dict['n_infected'] = len(next(iter(infected_cell_statistics[self.serum_key].values())))
        stat_dict['n_control'] = len(next(iter(control_cell_statistics[self.serum_key].values())))

        # this only accounts for cells that were either classified as infected or control
        stat_dict['n_cells'] = stat_dict['n_infected'] + stat_dict['n_control']
        stat_dict['fraction_infected'] = stat_dict['n_infected'] / stat_dict['n_cells']

        stat_dict['cell_size_median_infected'] = np.median(infected_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_mean_infected'] = np.mean(infected_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_median_control'] = np.median(control_cell_statistics[self.serum_key]['sizes'])
        stat_dict['cell_size_mean_control'] = np.mean(control_cell_statistics[self.serum_key]['sizes'])

        if bg_keys is not None:
            # the background statistics are saved for every cell, so get them from an arbitrary one
            all_cell_statistics = join_cell_properties(infected_cell_statistics, control_cell_statistics)
            for bg_key in bg_keys:
                for semantic_channel_name, channel_key in [('serum', self.serum_key), ('marker', self.marker_key)]:
                    try:
                        bg_stat = next(iter(all_cell_statistics[channel_key][bg_key]))
                    except StopIteration:
                        bg_stat = np.nan
                    stat_dict[f'{bg_key}_{semantic_channel_name}'] = bg_stat
        return stat_dict


class CellLevelAnalysisWithTableBase(CellLevelAnalysisBase):
    """
    """
    def __init__(self, table_out_keys, check_image_outputs=False, **super_kwargs):
        self.table_out_keys = table_out_keys
        self.check_image_outputs = check_image_outputs
        super().__init__(**super_kwargs)

    def check_table(self):
        table_path = self.table_out_path
        if not os.path.exists(table_path):
            return False
        with open_file(table_path, 'r') as f:
            if not all(key in f for key in self.table_out_keys):
                return False
        return True

    # we only write a single output file, so need to over-write the output validation and output checks
    def check_output(self, path):
        have_table = self.check_table()
        if self.check_image_outputs:
            return have_table and super().check_output(path)
        else:
            return have_table

    def validate_outputs(self, output_files, folder, status, ignore_failed_outputs):
        have_table = self.check_table()
        if self.check_image_outputs:
            return have_table and super().validate_outputs(output_files,
                                                           folder, status,
                                                           ignore_failed_outputs)
        else:
            return have_table


class CellLevelAnalysis(CellLevelAnalysisWithTableBase):
    """
    """
    # for now, we hard-code the table keys and write to different table files instead
    image_table_key = 'images/default'
    well_table_key = 'wells/default'

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 serum_bg_key='plate_bg_median',
                 marker_bg_key='plate_bg_median',
                 score_name='ratio_of_median_of_means',
                 write_summary_images=False,
                 infected_cell_mask_key='infected_cell_mask',
                 serum_per_cell_mean_key='serum_per_cell_mean',
                 edge_key='cell_segmentation_edges',
                 cell_outlier_table_name='outliers',
                 image_outlier_table='images/outliers',
                 **super_kwargs):

        self.score_name = score_name
        self.write_summary_images = write_summary_images

        self.image_outlier_table = image_outlier_table

        if self.write_summary_images:
            output_key = [infected_cell_mask_key,
                          serum_per_cell_mean_key,
                          edge_key]
            self.edge_key = edge_key
            self.infected_cell_mask_key = infected_cell_mask_key
            self.serum_per_cell_mean_key = serum_per_cell_mean_key
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
                         **super_kwargs)

        output_group = cell_seg_key if self.identifier is None else cell_seg_key + '_' + self.identifier
        self.cell_outlier_table = output_group + '/' + serum_key + '_' + cell_outlier_table_name

    def load_image_outliers(self, input_files):
        with open_file(self.table_out_path, 'r') as f:
            if not self.has_table(f, self.image_outlier_table):
                logger.warning(f"{self.name}: load_image_outliers: did not find an image outlier table")
                return {}
            keys, table = self.read_table(f, self.image_outlier_table)

        im_name_id = keys.index('image_name')
        outlier_id = keys.index('is_outlier')
        outlier_type_id = keys.index('outlier_type')

        image_names = set(table[:, im_name_id])
        expected_names = set(os.path.splitext(os.path.split(in_file)[1])[0]
                             for in_file in input_files)

        # TODO check if this actually works now
        if image_names != expected_names:
            msg = f"{self.name}: load_image_outliers: image names from table and expected image names do not agree"
            logger.warning(msg)

        outlier_dict = {table[ii, im_name_id]: (table[ii, outlier_id], table[ii, outlier_type_id])
                        for ii in range(len(table))}
        return outlier_dict

    def load_cell_outliers(self, input_file):
        with open_file(input_file, 'r') as f:
            if not self.has_table(f, self.cell_outlier_table):
                logger.warning(f"{self.name}: load_cell_outliers: did not find a cell outlier table")
                return {}
            keys, table = self.read_table(f, self.cell_outlier_table)

        label_id = keys.index('label_id')
        outlier_id = keys.index('is_outlier')
        outlier_type_id = keys.index('outlier_type')
        outlier_dict = {table[ii, label_id]: (table[ii, outlier_id], table[ii, outlier_type_id])
                        for ii in range(len(table))}
        return outlier_dict

    def write_image_table(self, input_files):

        image_outlier_dict = self.load_image_outliers(input_files)

        column_names = ['image_name', 'site_name', 'is_outlier', 'outlier_type', 'n_outlier_cells']
        table = []

        # TODO parallelize and tqdm
        for ii, in_file in enumerate(input_files):
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]

            infected_cell_statistics, control_cell_statistics = self.load_per_cell_statistics(in_file)

            # get all the statistics for this image and their names
            bg_keys = ['plate_bg_median', 'plate_bg_mad',
                       'well_bg_median', 'well_bg_mad',
                       'image_bg_median', 'image_bg_mad']
            stat_dict = self.get_stat_dict(infected_cell_statistics, control_cell_statistics, bg_keys)

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
            self.write_table(f, self.image_table_key, column_names, table, visible)

        return table, column_names

    def write_well_table(self, input_files):
        # group input files per well
        input_files_per_well = defaultdict(list)
        for in_file in input_files:
            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            well_name = image_name_to_well_name(image_name)
            input_files_per_well[well_name].append(in_file)

        image_outlier_dict = self.load_image_outliers(input_files)
        column_names = ['well_name', 'n_outlier_images', 'n_outlier_cells']
        table = []

        for ii, (well_name, in_files_for_current_well) in enumerate(input_files_per_well.items()):
            n_total = len(in_files_for_current_well)
            image_names_for_current_well = [os.path.splitext(os.path.split(in_file)[1])[0]
                                            for in_file in in_files_for_current_well]

            in_files_for_current_well = [in_file for in_file, im_name in zip(in_files_for_current_well,
                                                                             image_names_for_current_well)
                                         if not image_outlier_dict.get(im_name, (-1, ''))[0] == 1]
            n_outlier_images = n_total - len(in_files_for_current_well)
            if len(in_files_for_current_well) == 0:
                # TODO: add row full of np.nan for wells of outliers
                logger.info(f'Skipping well {well_name} as it consists entirely of outliers. '
                            f'It will not be included in the per-well table.')
                continue
            infected_cell_statistics, control_cell_statistics = self.load_per_cell_statistics(in_files_for_current_well)

            # get all the statistics for this well and their names
            bg_keys = ['plate_bg_median', 'plate_bg_mad',
                       'well_bg_median', 'well_bg_mad']
            stat_dict = self.get_stat_dict(infected_cell_statistics, control_cell_statistics, bg_keys)

            # get the main score, which is the measure computed for `score_name`, but set to
            # nan if this image is an outlier
            stat_dict['score'] = np.nan if stat_dict['score'] is None else stat_dict['score']

            stat_names, stat_list = map(list, zip(*stat_dict.items()))
            if ii == 0:
                column_names += list(stat_names)

            # get number of ignored outlier cells
            n_outlier_cells = sum(sum(1 for v in self.load_cell_outliers(in_file).values() if v[0] == 1)
                                  for in_file in in_files_for_current_well)

            table.append([well_name, n_outlier_images, n_outlier_cells] + stat_list)

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.well_table_key, column_names, table)

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

        # make a label mask for the infected cells
        label_ids = self.load_per_cell_statistics(in_path, False, False)['labels']
        infected_indicator, _ = self.load_infected_and_control_indicators(in_path)
        assert len(label_ids) == len(infected_indicator), f'{len(label_ids)} != {len(infected_indicator)}'
        infected_label_ids = label_ids[infected_indicator.astype('bool')]  # cast to bool again to be sure
        infected_mask = np.isin(cell_seg, infected_label_ids).astype(cell_seg.dtype)

        # TODO: should we subtract the background here?
        result = self.load_per_cell_statistics(in_path, subtract_background=True, split_infected_and_control=False)
        mean_serum_image = np.zeros_like(cell_seg, dtype=np.float32)
        for label, intensity in zip(filter(lambda x: x != 0, label_ids),
                                    result[self.serum_key]['means']):
            mean_serum_image[cell_seg == label] = intensity

        seg_edges = seg_to_edges(cell_seg).astype('uint8')

        with open_file(out_path, 'a') as f:
            # we need to use nearest down-sampling for the mean serum images,
            # because while these are float values, they should not be interpolated
            self.write_image(f, self.serum_per_cell_mean_key, mean_serum_image,
                             settings={'use_nearest': True})
            self.write_image(f, self.infected_cell_mask_key, infected_mask)
            self.write_image(f, self.edge_key, seg_edges)

    def run(self, input_files, output_files):
        image_table, image_columns = self.write_image_table(input_files)
        well_table, well_columns = self.write_well_table(input_files)
        self.write_image_and_well_information(output_files, image_table, image_columns,
                                              well_table, well_columns)

        if self.write_summary_images:
            for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                                desc='write cell level analysis summary images'):
                self.write_summary_image(input_file, output_file)
