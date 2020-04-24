import json
import os
import pickle
from copy import copy, deepcopy
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm

from batchlib.util.logging import get_logger
from ..base import BatchJobWithSubfolder
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


def substract_background_of_serum(cell_properties, bg_label=0):
    return substract_background_of_marker(cell_properties, marker_key='serum')


def divide_by_background_of_marker(cell_properties, bg_label=0):
    bg_ind = list(cell_properties['labels']).index(bg_label)
    cell_properties = deepcopy(cell_properties)
    mean_bg = cell_properties['marker']['means'][bg_ind]
    for key in cell_properties['marker'].keys():
        cell_properties['marker'][key] -= 550
        cell_properties['marker'][key] /= (mean_bg - 550)
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


def split_by_marker_threshold(cell_properties, threshold, statistic, marker_key='marker', return_infected_ind=False):
    # returns not infected, infected cell properties
    infected_ind = cell_properties[marker_key][statistic] > threshold
    not_infected_ind = 1 - infected_ind
    if not return_infected_ind:
        return index_cell_properties(cell_properties, not_infected_ind), \
               index_cell_properties(cell_properties, infected_ind)
    else:

        return index_cell_properties(cell_properties, not_infected_ind), \
               index_cell_properties(cell_properties, infected_ind), infected_ind


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


def compute_ratios(not_infected_properties, infected_properties):
    # input should be the return value of eval_cells
    not_infected_global_properties = compute_global_statistics(not_infected_properties)
    infected_global_properties = compute_global_statistics(infected_properties)
    result = dict()

    def serum_ratio(key, key2=None):
        key2 = key if key2 is None else key
        try:
            result = (infected_global_properties['serum'][key2]) / (not_infected_global_properties['serum'][key])
        except Exception:
            result = None
        return result

    def diff_over_sum(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties['serum'][key], not_infected_global_properties['serum'][key2]
            result = (inf - not_inf) / (inf + not_inf)
        except Exception:
            result = None
        return result

    def diff(key, key2=None):
        key2 = key if key2 is None else key
        try:
            inf, not_inf = infected_global_properties['serum'][key], not_infected_global_properties['serum'][key2]
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
    result['infected_mean'] = infected_global_properties['serum']['global_mean']
    result['infected_median'] = infected_global_properties['serum']['q0.5_of_cell_means']
    result['not_infected_mean'] = not_infected_global_properties['serum']['global_mean']
    result['not_infected_median'] = not_infected_global_properties['serum']['q0.5_of_cell_means']
    return result


def get_measures(cell_properties, infected_threshold, split_statistic='top50'):
    if isinstance(infected_threshold, (list, tuple, np.ndarray)):
        assert len(infected_threshold) == 2
        split = [
            split_by_marker_threshold(cell_properties, infected_threshold[0], split_statistic)[0],
            split_by_marker_threshold(cell_properties, infected_threshold[1], split_statistic)[1]
        ]
    else:
        split = split_by_marker_threshold(cell_properties, infected_threshold, split_statistic)
    return compute_ratios(*split)


class CellLevelAnalysis(BatchJobWithSubfolder):
    """
    """
    def __init__(self,
                 serum_key='serum',
                 marker_key='marker',
                 nuc_seg_key='nucleus_segmentation',
                 cell_seg_key='cell_segmentation',
                 output_folder='instancewise_analysis',
                 identifier=None):

        self.serum_key = serum_key
        self.marker_key = marker_key
        self.nuc_seg_key = nuc_seg_key
        self.cell_seg_key = cell_seg_key

        # all inputs should be 2d
        input_ndim = [2, 2, 2, 2]

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.pickle' if identifier is None else f'_{identifier}.pickle'

        super().__init__(output_ext=output_ext,
                         output_folder=output_folder,
                         input_key=[self.serum_key,
                                    self.marker_key,
                                    self.nuc_seg_key,
                                    self.cell_seg_key],
                         input_ndim=input_ndim,
                         identifier=identifier)

    def load_sample(self, path, device):
        with open_file(path, 'r') as f:
            serum = self.read_input(f, self.serum_key)
            marker = self.read_input(f, self.marker_key)
            nucleus_seg = self.read_input(f, self.nuc_seg_key)
            cell_seg = self.read_input(f, self.cell_seg_key)

        marker = torch.FloatTensor(marker.astype(np.float32)).to(device)
        serum = torch.FloatTensor(serum.astype(np.float32)).to(device)
        nucleus_seg = torch.LongTensor(nucleus_seg.astype(np.int32)).to(device)
        cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)

        cell_seg[nucleus_seg != 0] = 0

        return marker, serum, nucleus_seg, cell_seg

    def eval_cells(self, marker, serum, nucleus_seg, cell_seg,
                   ignore_label=0,
                   substract_mean_background=False):
        # all segs have shape H, W
        assert marker.shape == serum.shape == nucleus_seg.shape == cell_seg.shape
        # include background as instance with label 0
        labels = torch.sort(torch.unique(cell_seg))[0]

        if substract_mean_background:
            marker -= (marker[cell_seg == ignore_label]).mean()
            serum -= (serum[cell_seg == ignore_label]).mean()

        def get_per_mask_statistics(data):
            per_cell_values = [data[cell_seg == label] for label in labels]
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

        cell_properties = dict()
        cell_properties['marker'] = get_per_mask_statistics(marker)
        cell_properties['serum'] = get_per_mask_statistics(serum)
        cell_properties['labels'] = labels.cpu().numpy()

        return cell_properties

    # this is what should be run for each h5 file
    def save_all_stats(self, in_file, out_file, device):
        infected_threshold = 250  # TODO put this and other params into args of __init__
        split_statistic = 'top50'
        sample = self.load_sample(in_file, device=device)
        per_cell_statistics_to_save = self.eval_cells(*sample)

        per_cell_statistics = substract_background_of_marker(per_cell_statistics_to_save)
        per_cell_statistics = remove_background_of_cell_properties(per_cell_statistics)
        measures = get_measures(per_cell_statistics, infected_threshold, split_statistic=split_statistic)

        infected_ind = split_by_marker_threshold(per_cell_statistics, infected_threshold, split_statistic,
                                                 return_infected_ind=True)[-1]
        infected_ind_with_bg = np.zeros_like(per_cell_statistics_to_save['marker']['means'])
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

    def run(self, input_files, output_files, gpu_id=None):
        with torch.no_grad():
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                device = torch.device(0)
            else:
                device = torch.device('cpu')

            _save_all_stats = partial(self.save_all_stats, device=device)
            for input_file, output_file in tqdm(list(zip(input_files, output_files)),
                                                desc='running cell-level analysis on images'):
                _save_all_stats(input_file, output_file)
