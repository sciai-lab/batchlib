from tqdm.auto import tqdm
import torch
import numpy as np
import h5py
import pickle
from functools import partial
from concurrent import futures
from ..base import BatchJob


def load_sample(path, cell_seg_key, device='cpu'):
    with h5py.File(path, 'r') as f:
        raw = f['raw'][()]
        nucleus_seg = f['nucleus_segmentation'][()][None]
        cell_seg = f[cell_seg_key][()][None]
        fg_mask = f['foreground_mask'][()][None]
    gfp = raw[1][None]
    serum = raw[2][None]

    gfp = torch.FloatTensor(gfp.astype(np.float32)).to(device)
    serum = torch.FloatTensor(serum.astype(np.float32)).to(device)
    nucleus_seg = torch.LongTensor(nucleus_seg.astype(np.int32)).to(device)
    cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)
    fg_mask = torch.LongTensor(fg_mask.astype(np.int32)).to(device)

    cell_seg[nucleus_seg != 0] = 0

    return fg_mask, gfp, serum, nucleus_seg, cell_seg


def eval_cells(fg_mask, gfp, serum, nucleus_seg, cell_seg, ignore_label=0,
               substract_mean_background=False):
    # all segs have shape C=1, H, W
    assert gfp.shape[0] == serum.shape[0] == nucleus_seg.shape[0] == cell_seg.shape[0] == 1
    labels = torch.unique(cell_seg[cell_seg != ignore_label])

    if substract_mean_background:
        gfp -= (gfp[cell_seg == ignore_label]).mean()
        serum -= (serum[cell_seg == ignore_label]).mean()

    def get_per_mask_statistics(data):
        per_cell_values = [data[cell_seg == label] for label in labels]
        sums = torch.Tensor([arr.sum() for arr in per_cell_values])
        means = torch.Tensor([arr.mean() for arr in per_cell_values])
        instance_sizes = torch.Tensor([len(arr.view(-1)) for arr in per_cell_values])

        top50 = torch.Tensor([0 if len(t) < 50 else t.topk(50)[0][-1]
                             for t in per_cell_values])
        top30 = torch.Tensor([0 if len(t) < 30 else t.topk(30)[0][-1]
                              for t in per_cell_values])
        top10 = torch.Tensor([0 if len(t) < 10 else t.topk(10)[0][-1]
                             for t in per_cell_values])
        # convert to numpy here
        return dict(sums=sums.cpu().numpy(),
                    means=means.cpu().numpy(),
                    sizes=instance_sizes.cpu().numpy(),
                    top50=top50.cpu().numpy(),
                    top30=top30.cpu().numpy(),
                    top10=top10.cpu().numpy())

    cell_properties = dict()
    cell_properties['gfp'] = get_per_mask_statistics(gfp)
    cell_properties['serum'] = get_per_mask_statistics(serum)

    return cell_properties


# this is what should be run for each h5 file
def save_all_stats(in_file, out_file, cell_seg_key):
    sample = load_sample(in_file, cell_seg_key)
    per_cell_statistics = eval_cells(*sample)
    with open(out_file, 'wb') as f:
        pickle.dump(per_cell_statistics, f)


class CellLevelAnalysis(BatchJob):
    """
    """
    def __init__(self, input_pattern='*.h5', mode='ws', cell_seg_key=None):
        self.mode = mode
        self.cell_seg_key = cell_seg_key if cell_seg_key is not None else 'cell_segmentation_%s' % self.mode
        super().__init__(input_pattern,
                         output_ext=f'_{mode}.pickle',
                         input_key=['raw', 'nucleus_segmentation', 'foreground_mask', self.cell_seg_key])
        self.runners = {'default': self.run}

    def run(self, input_files, output_files, n_jobs=1):
        _save_all_stats = partial(save_all_stats, cell_seg_key=self.cell_seg_key)

        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_save_all_stats, input_files, output_files), total=len(input_files)))
