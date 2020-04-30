import os
from copy import copy, deepcopy
from functools import partial

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
    # TODO: also save per-well (even per plate?) values of BG in table
    def __init__(self,
                 channel_keys=('serum', 'marker'),
                 nuc_seg_key='nucleus_segmentation',
                 cell_seg_key='cell_segmentation',
                 identifier=None):

        self.channel_keys = tuple(channel_keys)
        self.nuc_seg_key = nuc_seg_key
        self.cell_seg_key = cell_seg_key

        # all inputs should be 2d
        input_ndim = [2, 2, 2, 2]

        # tables are per default saved at tables/cell_segmentation/channel in the container
        output_group = cell_seg_key if identifier is None else cell_seg_key + '_' + identifier
        self.output_table_keys = [output_group + '/' + channel for channel in channel_keys]
        super().__init__(input_key=list(self.channel_keys + (self.nuc_seg_key, self.cell_seg_key)),
                         input_ndim=input_ndim,
                         output_key=['tables/' + key for key in self.output_table_keys],
                         identifier=identifier)

    def load_sample(self, path, device):
        with open_file(path, 'r') as f:
            channels = [self.read_image(f, key) for key in self.channel_keys]
            nucleus_seg = self.read_image(f, self.nuc_seg_key)
            cell_seg = self.read_image(f, self.cell_seg_key)

        channels = [torch.FloatTensor(channel.astype(np.float32)).to(device) for channel in channels]
        nucleus_seg = torch.LongTensor(nucleus_seg.astype(np.int32)).to(device)
        cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)

        cell_seg[nucleus_seg != 0] = 0

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
                   ignore_label=0,
                   substract_mean_background=False):
        # all segs have shape H, W
        shape = cell_seg.shape
        for channel in list(channels):
            assert channel.shape == shape

        # include background as instance with label 0
        labels = torch.sort(torch.unique(cell_seg))[0]

        if substract_mean_background:
            for channel in channels:
                channel -= (channel[cell_seg == ignore_label]).mean()

        cell_properties = dict()
        for key, channel in zip(self.channel_keys, channels):
            cell_properties[key] = self.get_per_instance_statistics(channel, cell_seg, labels)
        cell_properties['labels'] = labels.cpu().numpy()

        return cell_properties

    # this is what should be run for each h5 file
    def save_all_stats(self, in_file, out_file, device):
        sample = self.load_sample(in_file, device=device)
        per_cell_statistics = self.eval_cells(*sample)

        labels = per_cell_statistics['labels']
        for channel, output_key in zip(self.channel_keys, self.output_table_keys):
            columns = ['label_id']
            table = [list(labels)]
            for key, values in per_cell_statistics[channel].items():
                columns.append(key)
                table.append([v if v is not None else np.nan for v in values])
            # transpose table to have shape (n_cells, n_features)
            table = np.asarray(table, dtype=float).T
            with open_file(out_file, 'a') as f:
                self.write_table(f, output_key, columns, table)

    def run(self, input_files, output_files, gpu_id=None):
        with torch.no_grad():
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                device = torch.device(0)
            else:
                device = torch.device('cpu')
            _save_all_stats = partial(self.save_all_stats, device=device)
            for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                                desc='extracting cell-level features'):
                _save_all_stats(input_file, output_file)


class FindInfectedCells(BatchJobOnContainer):
    """This job finds the infected and control cells to later compute the measures on"""
    def __init__(self,
                 cell_seg_key='cell_segmentation',
                 marker_key='marker',
                 infected_threshold=250, split_statistic='top50',
                 bg_correction_key=None,
                 identifier=None,
                 **super_kwargs):
        self.marker_key = marker_key
        self.cell_seg_key = cell_seg_key
        self.feature_table_key = cell_seg_key + '/' + marker_key

        self.infected_threshold = infected_threshold
        self.split_statistic = split_statistic

        self.bg_correction_key = bg_correction_key

        # infected are per default saved at tables/infected_ind/cell_segmentation/marker_key in the container
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

    def get_infected_ind(self, feature_dict):
        bg_ind = feature_dict['label_id'].tolist().index(0)
        if self.bg_correction_key:
            offset = feature_dict[self.bg_correction_key][bg_ind]
        else:
            offset = 0
        infected_ind = feature_dict[self.split_statistic] > self.infected_threshold + offset
        infected_ind[bg_ind] = False  # the background should never be classified as infected
        return infected_ind

    def get_infected_and_control_ind(self, feature_dict):
        infected_ind = self.get_infected_ind(feature_dict)
        # per default, everything that is not infected is control
        bg_ind = feature_dict['label_id'].tolist().index(0)
        control_ind = infected_ind == False
        control_ind[bg_ind] = False  # the background should never be classified as control
        return infected_ind, control_ind

    def compute_and_save_infected_ind(self, in_file, out_file):
        feature_dict = self.load_feature_dict(in_file)
        infected_ind, control_ind = self.get_infected_and_control_ind(feature_dict)
        column_names = ['label_id', 'is_infected', 'is_control']
        table = [feature_dict['label_id'], infected_ind, control_ind]
        table = np.asarray(table, dtype=float).T
        with open_file(out_file, 'a') as f:
            self.write_table(f, self.output_table_key, column_names, table)

    def run(self, input_files, output_files):
        for input_file, output_file in tqdm(list(zip(input_files, output_files)), desc='finding infected cells'):
            self.compute_and_save_infected_ind(input_file, output_file)


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
    def folder_to_table_path(folder):
        # NOTE, we call this .hdf5 to avoid pattern matching, it's a bit hacky ...
        table_file_name = os.path.split(folder)[1] + '_table.hdf5'
        return os.path.join(folder, table_file_name)

    @property
    def table_out_path(self):
        return self.folder_to_table_path(self.folder)

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

    def load_infected_and_control_ind(self, in_path):
        with open_file(in_path, 'r') as f:
            column_names, table = self.read_table(f, self.classification_key)
        infected_ind = table[:, 1]
        control_ind = table[:, 2]
        return infected_ind, control_ind

    # this is what should be run for each h5 file
    def write_image_table(self, input_files):

        column_names = ['image_name', 'site_name', 'score', 'marked_as_outlier', 'n_infected', 'n_control']
        table = []

        for ii, in_file in enumerate(input_files):
            per_cell_statistics_to_save = self.load_result(in_file)
            per_cell_statistics = deepcopy(per_cell_statistics_to_save)  # TODO: background subtraction

            infected_ind, control_ind = self.load_infected_and_control_ind(in_file)
            n_infected = infected_ind.astype(np.int32).sum()
            n_control = control_ind.astype(np.int32).sum()

            control_cell_statistics = index_cell_properties(per_cell_statistics, control_ind)
            infected_cell_statistics = index_cell_properties(per_cell_statistics, infected_ind)
            measures = compute_ratios(control_cell_statistics, infected_cell_statistics, serum_key=self.serum_key)
            if ii == 0:
                column_names += list(measures.keys())

            image_name = os.path.splitext(os.path.split(in_file)[1])[0]
            site_name = image_name_to_site_name(image_name)

            # outliers can have the following values:
            # 0: not an outlier
            # 1: outlier
            # -1: no annotation available
            outlier = self.outlier_predicate(image_name)
            score = measures[self.score_name]
            score = np.nan if (outlier == 1 or score is None) else score
            table.append([image_name, site_name, score, outlier, n_infected, n_control] +
                         [np.nan if m is None else m for m in measures.values()])

        # NOTE: todos left from Roman
        # TODO: Make one big per-cell table for the analysis, with both serum and marker statistics?

        table = np.array(table)
        n_cols = len(column_names)
        assert n_cols == table.shape[1]

        # set image name to non-visible for the plateViewer (something else?)
        visible = np.ones(n_cols, dtype='bool')
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
        well_column_names = ['well_name'] + column_names

        table = []
        # could maybe be done more efficiently
        for well_name in unique_wells:
            well_mask = well_names == well_name
            this_values = value_table[well_mask]

            row = []
            for col_id, name in enumerate(column_names):
                # TODO we should take median by default, but use
                # different rules for special names.
                # e.g. for number of cells we should use sum
                # for outliers, we should count the number of outliers etc.
                try:
                    row_values = this_values[:, col_id].astype('float')
                    this_value = np.median(row_values[np.isfinite(row_values)])
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

        # TODO this needs to be simplified
        # result = self.load_result(in_path)
        # labels = result['per_cell_statistics']['labels']
        # labels = np.array([], dtype=np.int32) if labels is None else labels
        # infected_labels = labels[result['infected_ind'] != 0]

        # infected_mask = np.zeros_like(cell_seg)
        # for label in infected_labels:
        #     infected_mask[cell_seg == label] = 1

        # mean_serum_image = np.zeros_like(cell_seg, dtype=np.float32)
        # for label, intensity in zip(filter(lambda x: x != 0, labels),
        #                             result['per_cell_statistics'][self.serum_key]['means']):
        #     mean_serum_image[cell_seg == label] = intensity

        seg_edges = seg_to_edges(cell_seg).astype('uint8')

        with open_file(out_path, 'a') as f:
            # we need to use nearest down-sampling for the mean serum images,
            # because while these are float values, they should not be interpolated
            # self.write_image(f, self.serum_per_cell_mean_key, mean_serum_image,
            #                  settings={'use_nearest': True})
            # self.write_image(f, self.infected_cell_mask_key, infected_mask)
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
