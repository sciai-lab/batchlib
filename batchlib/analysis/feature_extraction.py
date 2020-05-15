import os
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from ..base import BatchJobOnContainer
from ..util import open_file


# TODO compute intensity independent properties like size
# and anchors (= region center points)
class SegmentationProperties(BatchJobOnContainer):
    pass


class InstanceFeatureExtraction(BatchJobOnContainer):
    def __init__(self,
                 channel_keys=('serum', 'marker'),
                 nuc_seg_key_to_ignore=None,
                 cell_seg_key='cell_segmentation',
                 topk=(10, 30, 50),
                 quantiles=tuple(),
                 identifier=None):

        self.channel_keys = tuple(channel_keys)
        self.nuc_seg_key_to_ignore = nuc_seg_key_to_ignore
        self.cell_seg_key = cell_seg_key

        self.topk = topk
        self.quantiles = quantiles

        # all inputs should be 2d
        input_ndim = [2] * (1 + len(channel_keys) + (1 if nuc_seg_key_to_ignore else 0))

        # tables are per default saved at tables/cell_segmentation/channel in the container
        output_group = cell_seg_key if identifier is None else cell_seg_key + '_' + identifier
        self.output_table_keys = [output_group + '/' + channel for channel in channel_keys]
        super().__init__(input_key=list(self.channel_keys) + [self.cell_seg_key] +
                                       ([self.nuc_seg_key_to_ignore] if self.nuc_seg_key_to_ignore is not None else []),
                         input_ndim=input_ndim,
                         output_key=self.output_table_keys,
                         output_format=['table'] * len(self.output_table_keys),
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
        topkdict = {f'top{k}': np.array([0 if len(t) < k else t.topk(k)[0][-1].item()
                                         for t in per_cell_values])
                    for k in self.topk}
        quantile_dict = {f'quantile{q}': np.array([np.quantile(t.cpu().numpy(), q)
                                                   for t in per_cell_values])
                         for q in self.quantiles}
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
                    **topkdict,
                    **quantile_dict)

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

    # this is what should be run for each h5 file
    def save_all_stats(self, in_file, out_file, device):
        sample = self.load_sample(in_file, device=device)
        per_cell_statistics = self.eval_cells(*sample)

        labels = per_cell_statistics['labels']
        for i, (channel, output_key) in enumerate(zip(self.channel_keys, self.output_table_keys)):
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
