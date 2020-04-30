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


def remove_background_of_cell_properties(cell_properties, bg_label=0):
    return index_cell_properties(cell_properties, np.array(cell_properties['labels']) != bg_label)


def substract_background_of_marker(cell_properties, bg_label=0, marker_key='marker'):
    bg_ind = list(cell_properties['labels']).index(bg_label)
    assert bg_ind is not None
    cell_properties = deepcopy(cell_properties)
    mean_bg = cell_properties[marker_key]['means'][bg_ind]
    for key in cell_properties[marker_key].keys():
        cell_properties[marker_key][key] -= mean_bg
    return cell_properties


def substract_background_of_serum(cell_properties, bg_label=0, serum_key='serum'):
    return substract_background_of_marker(cell_properties, marker_key=serum_key)


def divide_by_background_of_marker(cell_properties, marker_key, bg_label=0):
    bg_ind = list(cell_properties['labels']).index(bg_label)
    cell_properties = deepcopy(cell_properties)
    mean_bg = cell_properties[marker_key]['means'][bg_ind]
    for key in cell_properties[marker_key].keys():
        cell_properties[marker_key][key] -= 550
        cell_properties[marker_key][key] /= (mean_bg - 550)
    return cell_properties


def join_cell_properties(*cell_property_list):
    updated_cell_properties = []
    for cell_properties in cell_property_list:
        cell_properties = copy(cell_properties)
        cell_properties.pop('measures', None)
        cell_properties.pop('labels', None)
        updated_cell_properties.append(cell_properties)
    cell_property_list = updated_cell_properties
    return {key: {inner_key: np.concatenate([cell_property[key][inner_key]
                                             for cell_property in cell_property_list
                                             if len(cell_property[key][inner_key].shape) > 0])
                  for inner_key in value.keys()} if isinstance(value, dict) else None
            for key, value in cell_property_list[0].items()}


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
                result = None
            return result

        result[channel]['q0.5_of_cell_sums'] = robust_quantile(properties['sums'], 0.5)
        result[channel]['q0.5_of_cell_means'] = robust_quantile(properties['means'], 0.5)
        result[channel]['q0.3_of_cell_means'] = robust_quantile(properties['means'], 0.3)
        result[channel]['q0.7_of_cell_means'] = robust_quantile(properties['means'], 0.7)
        result[channel]['q0.1_of_cell_means'] = robust_quantile(properties['means'], 0.1)
        result[channel]['q0.9_of_cell_means'] = robust_quantile(properties['means'], 0.9)
        result[channel]['cell_mean'] = properties['means'].mean()
        result[channel]['cell_sum'] = properties['sums'].mean()
    return result


def compute_ratios(not_infected_properties, infected_properties, serum_key='serum'):
    # input should be the return value of eval_cells
    not_infected_global_properties = compute_global_statistics(not_infected_properties)
    infected_global_properties = compute_global_statistics(infected_properties)
    result = dict()

    def serum_ratio(key, key2=None):
        key2 = key if key2 is None else key
        try:
            result = (infected_global_properties[serum_key][key2]) / (not_infected_global_properties[serum_key][key])
        except Exception:
            result = None
        return result

    def diff_over_sum(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties[serum_key][key], not_infected_global_properties[serum_key][key2]
            result = (inf - not_inf) / (inf + not_inf)
        except Exception:
            result = None
        return result

    def diff(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties[serum_key][key], not_infected_global_properties[serum_key][key2]
            result = inf - not_inf
        except Exception:
            result = None
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
    for key, value in not_infected_global_properties[serum_key].items():
        result[f'not_infected_{key}'] = value

    result['infected_mean'] = infected_global_properties[serum_key]['global_mean']
    result['infected_median'] = infected_global_properties[serum_key]['q0.5_of_cell_means']
    result['not_infected_mean'] = not_infected_global_properties[serum_key]['global_mean']
    result['not_infected_median'] = not_infected_global_properties[serum_key]['q0.5_of_cell_means']
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
        medians = bg_values.median(1)[0]
        mads = (bg_values - medians[:, None]).abs().median(1)[0]

        def to_numpy(tensor):
            return tensor.cpu().numpy().astype(np.float32)
        return {
            'median': to_numpy(medians),
            'mad': to_numpy(mads),
        }

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
                bg_ind = feature_dict['label_id'].tolist().index(0)
                offset = feature_dict[bg_correction_key][bg_ind]
        else:
            offset = 0
        return offset

    def get_infected_indicator(self, feature_dict):
        bg_ind = feature_dict['label_id'].tolist().index(0)
        offset = self.get_bg_correction(feature_dict)
        if self.infected_threshold_scale_key is not None:
            scale = feature_dict[self.infected_threshold_scale_key]
        else:
            scale = 1
        infected_indicator = feature_dict[self.split_statistic] > scale * self.infected_threshold + offset
        infected_indicator[bg_ind] = False  # the background should never be classified as infected
        return infected_indicator

    def get_infected_and_control_indicators(self, feature_dict):
        infected_indicator = self.get_infected_indicator(feature_dict)
        # per default, everything that is not infected is control
        bg_ind = feature_dict['label_id'].tolist().index(0)
        control_indicator = infected_indicator == False
        control_indicator[bg_ind] = False  # the background should never be classified as control
        return infected_indicator, control_indicator

    def compute_and_save_infected_and_control(self, in_file, out_file):
        feature_dict = self.load_feature_dict(in_file)
        infected_indicator, control_indicator= self.get_infected_and_control_indicators(feature_dict)
        column_names = ['label_id', 'is_infected', 'is_control']
        table = [feature_dict['label_id'], infected_indicator, control_indicator]
        table = np.asarray(table, dtype=float).T
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_table_key, column_names, table)

    def run(self, input_files, output_files):
        for input_file, output_file in tqdm(list(zip(input_files, output_files)), desc='finding infected cells'):
            self.compute_and_save_infected_and_control(input_file, output_file)


class CellLevelAnalysis(BatchJobOnContainer):
    """
    """
    # TODO enable over-riding these keys to allow runnning CellLevelAnalysis Jobs
    # with different settings on the same folder
    image_table_key = 'images/default'
    well_table_key = 'wells/default'

    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 serum_key='serum',
                 marker_key='marker',
                 outlier_predicate=lambda im: False,
                 score_name='ratio_of_median_of_means',
                 infected_threshold=250, split_statistic='top50',
                 subtract_marker_background=True,
                 write_summary_images=False,
                 infected_cell_mask_key='infected_cell_mask',
                 serum_per_cell_mean_key='serum_per_cell_mean',
                 edge_key='cell_segmentation_edges',
                 **super_kwargs):

        self.outlier_predicate = outlier_predicate

        # TODO allow for serum and marker data to come from different segmentations
        self.serum_key = cell_seg_key + '/' + serum_key
        self.marker_key = cell_seg_key + '/' + marker_key
        self.classification_key = 'cell_classification/' + cell_seg_key + '/' + marker_key

        self.infected_threshold = infected_threshold
        self.split_statistic = split_statistic

        self.subtract_marker_background = subtract_marker_background
        self.score_name = score_name
        self.write_summary_images = write_summary_images

        if self.write_summary_images:
            output_key = [infected_cell_mask_key,
                          serum_per_cell_mean_key,
                          edge_key]
            self.edge_key = edge_key
            self.infected_cell_mask_key = infected_cell_mask_key
            self.serum_per_cell_mean_key = serum_per_cell_mean_key
        else:
            output_key = None

        self.cell_seg_key = cell_seg_key
        super().__init__(input_key=[f'tables/{key}'
                                    for key in (self.serum_key, self.marker_key, self.classification_key)],
                         output_key=output_key,
                         **super_kwargs)

    # in the long run we should merge this into BatchJobOnContainer somehow
    def validate_input(self, path):
        if not os.path.exists(path):
            return False

        exp_keys = self._input_exp_key
        if exp_keys is None:
            return True
        with open_file(path, 'r') as f:
            for key in exp_keys:
                if key not in f:
                    print("111", key)
                    return False
                g = f[key]
                if ('cells' not in g) or ('columns' not in g):
                    print('AAA', key)
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

    def check_table(self):
        table_path = self.table_out_path
        if not os.path.exists(table_path):
            return False
        with open_file(table_path, 'r') as f:
            if self.image_table_key not in f:
                return False
            if self.well_table_key not in f:
                return False
        return True

    # we only write a single output file, so need to over-write the output validation and output checks
    def check_output(self, path):
        have_table = self.check_table()
        if self.write_summary_images:
            return have_table and super().check_output(path)
        else:
            return have_table

    def validate_outputs(self, output_files, folder, status, ignore_failed_outputs):
        have_table = self.check_table()
        if self.write_summary_images:
            return have_table and super().validate_outputs(output_files,
                                                           folder, status,
                                                           ignore_failed_outputs)
        else:
            return have_table

    def load_result(self, in_path):
        with open_file(in_path, 'r') as f:
            serum_keys, serum_table = self.read_table(f, self.serum_key)
            serum_dict = {key: values for key, values in zip(serum_keys, serum_table.T)}
            marker_keys, marker_table = self.read_table(f, self.marker_key)
            marker_dict = {key: values for key, values in zip(marker_keys, marker_table.T)}
        assert np.all(serum_dict['label_id'] == marker_dict['label_id'])
        return {
            'labels': serum_dict['label_id'],
            self.serum_key: serum_dict,
            self.marker_key: marker_dict
        }

    def load_infected_and_control_indicators(self, in_path):
        with open_file(in_path, 'r') as f:
            column_names, table = self.read_table(f, self.classification_key)
        infected_indicator = table[:, 1]
        control_indicator = table[:, 2]
        return infected_indicator, control_indicator

    # TODO for now, we only use manual outlier annotations, but we should
    # also use some heuristics for automated QC, e.g.
    # - number of cells
    # - cell size distribution
    # - negative ratios
    def check_for_outlier(self, image_name, values, value_names):

        # outliers can have the following values:
        # 0: not an outlier
        # 1: outlier
        # -1: no annotation available
        outlier = self.outlier_predicate(image_name)

        outlier_type = 'none'
        if outlier == 1:
            outlier_type = 'manual'
        if outlier == -1:
            outlier_type = 'not annotated'

        return outlier, outlier_type

    # this is what should be run for each h5 file
    def write_image_table(self, input_files):

        column_names = ['image_name', 'site_name', 'score', 'marked_as_outlier', 'outlier_type',
                        'n_infected', 'n_control']
        table = []

        for ii, in_file in enumerate(input_files):
            per_cell_statistics_to_save = self.load_result(in_file)
            per_cell_statistics = deepcopy(per_cell_statistics_to_save)  # TODO: background subtraction

            # TODO: @Roman are these binary masks or indices?
            # Could we rename the variables to something that makes this clear?
            # (ind could be either 'indicator' (=binary mask) or index)
            infected_indicator, control_indicator = self.load_infected_and_control_indicators(in_file)
            n_infected = infected_indicator.astype(np.int32).sum()
            n_control = control_indicator.astype(np.int32).sum()

            control_cell_statistics = index_cell_properties(per_cell_statistics, control_indicator)
            infected_cell_statistics = index_cell_properties(per_cell_statistics, infected_indicator)
            measures = compute_ratios(control_cell_statistics, infected_cell_statistics, serum_key=self.serum_key)
            if ii == 0:
                column_names += list(measures.keys())

            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            site_name = image_name_to_site_name(image_name)

            # gather the values (and their names) that could be relevant for the outlier detection
            image_values = [n_infected, n_control] + [np.nan if m is None else m for m in measures.values()]
            value_names = column_names[5:]
            assert len(value_names) == len(image_values)

            # check if this image is an outlier. it can be classified as outlier either because
            # of outlier annotations or because it doesn't pass some quality control heuristics
            outlier, outlier_type = self.check_for_outlier(image_name, image_values, value_names)

            # get the main score, which is the measure computed for `score_name`, but set to
            # nan if this image is an outlier
            score = measures[self.score_name]
            score = np.nan if (outlier == 1 or score is None) else score
            table.append([image_name, site_name, score, outlier, outlier_type] + image_values)

        # NOTE: todos left from Roman
        # TODO: Make one big per-cell table for the analysis, with both serum and marker statistics?

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        # set image name to non-visible for the plateViewer (something else?)
        visible = np.ones(n_cols, dtype='uint8')
        visible[0] = False

        with open_file(self.table_out_path, 'a') as f:
            self.write_table(f, self.image_table_key, column_names, table, visible)

        return table, column_names

    def write_well_table(self, input_files, image_table, image_columns):
        site_names = image_table[:, 1]
        well_names = np.array([name.split('-')[0] for name in site_names])

        value_table = image_table[:, 2:]
        column_names = image_columns[2:]
        assert len(column_names) == value_table.shape[1]

        unique_wells = np.unique(well_names)
        well_column_names = ['well_name'] + [name for name in column_names if name != 'outlier_type']

        table = []
        # could maybe be done more efficiently
        for well_name in unique_wells:
            well_mask = well_names == well_name
            this_values = value_table[well_mask]

            row = []
            for col_id, name in enumerate(column_names):

                # we accumulate values over images with the median by default,
                # but use sum for the outliers and the number of cells
                # TODO are there other stats that need to be treated differently?
                if name in ('marked_as_outlier', 'n_control', 'n_infected'):
                    accumulator = np.sum
                elif name == 'outlier_type':
                    # we don't accumulate the outlier type column
                    continue
                else:
                    accumulator = np.median

                try:
                    row_values = this_values[:, col_id].astype('float')
                    this_value = accumulator(row_values[np.isfinite(row_values)])
                except ValueError:
                    this_value = np.nan
                row.append(this_value)

            table.append([well_name] + row)

        table = np.array(table)
        n_cols = len(well_column_names)

        assert table.shape[1] == n_cols

        return table, well_column_names

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
                                    well_information=str(well_information[well_name]))

    def write_summary_image(self, in_path, out_path):

        with open_file(in_path, 'r') as f:
            cell_seg = self.read_image(f, self.cell_seg_key)

        # make a label mask for the infected cells
        label_ids = np.unique(cell_seg)
        infected_indicator, _ = self.load_infected_and_control_indicators(in_path)
        assert len(label_ids) == len(infected_indicator)
        infected_label_ids = label_ids[infected_indicator.astype('bool')]  # cast to bool again to be sure
        infected_mask = np.isin(cell_seg, infected_label_ids).astype(cell_seg.dtype)

        result = self.load_result(in_path)
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
        well_table, well_columns = self.write_well_table(input_files, image_table, image_columns)
        self.write_image_and_well_information(output_files, image_table, image_columns,
                                              well_table, well_columns)

        if self.write_summary_images:
            for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                                desc='write cell level analysis summary images'):
                self.write_summary_image(input_file, output_file)
