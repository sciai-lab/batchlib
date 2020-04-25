import json
import os
import pickle
from copy import copy, deepcopy
from functools import partial

import numpy as np
import skimage.morphology
import torch
from tqdm.auto import tqdm

from batchlib.util.logging import get_logger
from ..base import BatchJobWithSubfolder, BatchJobOnContainer
from ..config import get_default_extension
from ..util.io import open_file

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
                img = self.read_input(f, self.input_key)
            img = self.denoise(img)
            with open_file(output_file, 'a') as f:
                self.write_result(f, self.output_key, img)


class DenoiseByGrayscaleOpening(DenoiseChannel):
    def __init__(self, radius=5, **super_kwargs):
        super(DenoiseByGrayscaleOpening, self).__init__(**super_kwargs)
        self.structuring_element = skimage.morphology.disk(radius)

    def denoise(self, img):
        return skimage.morphology.opening(img, selem=self.structuring_element)


class InstanceFeatureExtraction(BatchJobWithSubfolder):
    def __init__(self,
                 channel_keys=('serum', 'marker'),
                 nuc_seg_key='nucleus_segmentation',
                 cell_seg_key='cell_segmentation',
                 output_folder='instancewise_analysis'):

        self.channel_keys = tuple(channel_keys)
        self.nuc_seg_key = nuc_seg_key
        self.cell_seg_key = cell_seg_key

        # all inputs should be 2d
        input_ndim = [2, 2, 2, 2]

        # identifier allows to run different instances of this job on the same folder
        output_ext = '_features.pickle'

        super().__init__(output_ext=output_ext,
                         output_folder=output_folder,
                         input_key=list(self.channel_keys + (self.nuc_seg_key, self.cell_seg_key)),
                         input_ndim=input_ndim)

    def load_sample(self, path, device):
        with open_file(path, 'r') as f:
            channels = [self.read_input(f, key) for key in self.channel_keys]
            nucleus_seg = self.read_input(f, self.nuc_seg_key)
            cell_seg = self.read_input(f, self.cell_seg_key)

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
        # convert to numpy here
        return dict(sums=sums.cpu().numpy(),
                    means=means.cpu().numpy(),
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

        with open(out_file, 'wb') as f:
            pickle.dump(per_cell_statistics, f)

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


class CellLevelAnalysis(BatchJobWithSubfolder):
    """
    """
    def __init__(self,
                 serum_key='serum', marker_key='marker',
                 output_folder='instancewise_analysis',
                 infected_threshold=250, split_statistic='top50',
                 identifier=None):
        self.serum_key = serum_key
        self.marker_key = marker_key
        self.infected_threshold = infected_threshold
        self.split_statistic = split_statistic

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.pickle' if identifier is None else f'_{identifier}.pickle'

        super().__init__(output_ext=output_ext,
                         output_folder=output_folder,
                         identifier=identifier)

    def load_result(self, in_path):
        ext = get_default_extension()
        assert in_path.endswith(ext)
        # load result of cell level feature extraction
        split_path = os.path.abspath(in_path).split(os.sep)
        result_path = os.path.join('/', *split_path[:-1], self.output_folder, split_path[-1][:-3] + '_features.pickle')
        assert os.path.isfile(result_path), f'Cell feature file missing: {result_path}'
        with open(result_path, 'rb') as f:
            return pickle.load(f)

    def preprocess_per_cell_statistics(self, per_cell_statistics):
        per_cell_statistics = substract_background_of_marker(per_cell_statistics, marker_key=self.marker_key)
        per_cell_statistics = remove_background_of_cell_properties(per_cell_statistics)
        return per_cell_statistics

    def get_infected_ind(self, per_cell_statistics):
        return per_cell_statistics[self.marker_key][self.split_statistic] > self.infected_threshold

    # this is what should be run for each h5 file
    def save_all_stats(self, in_file, out_file):
        per_cell_statistics_to_save = self.load_result(in_file)
        per_cell_statistics = self.preprocess_per_cell_statistics(deepcopy(per_cell_statistics_to_save))

        infected_ind = self.get_infected_ind(per_cell_statistics)

        not_infected_ind = 1 - infected_ind
        not_infected_cell_statistics = index_cell_properties(per_cell_statistics, not_infected_ind)
        infected_cell_statistics = index_cell_properties(per_cell_statistics, infected_ind)
        measures = compute_ratios(not_infected_cell_statistics, infected_cell_statistics, serum_key=self.serum_key)

        infected_ind_with_bg = np.zeros_like(per_cell_statistics_to_save[self.marker_key]['means'])
        infected_ind_with_bg[per_cell_statistics_to_save['labels'] > 0] = infected_ind

        result = dict(per_cell_statistics=per_cell_statistics_to_save,
                      infected_ind=infected_ind_with_bg,
                      measures=measures)
        with open(out_file, 'wb') as f:
            pickle.dump(result, f)

        # also save the measures in jsons
        measures = {key: (float(value) if (value is not None and np.isreal(value)) else None)
                    for key, value in result['measures'].items()}
        with open(out_file[:-6] + 'json', 'w') as fp:
            json.dump(measures, fp)

    def run(self, input_files, output_files):
        for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                            desc='running cell-level analysis on images'):
            self.save_all_stats(input_file, output_file)
