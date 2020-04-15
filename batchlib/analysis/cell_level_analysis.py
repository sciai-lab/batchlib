from tqdm.auto import tqdm
import torch
import numpy as np
import pickle
import os
from functools import partial

from ..util.io import open_file
from ..base import BatchJobOnContainer


class CellLevelAnalysis(BatchJobOnContainer):
    """
    """
    def __init__(self,
                 raw_key='raw',
                 nuc_seg_key='nucleus_segmentation',
                 cell_seg_key='cell_segmentation',
                 identifier=None, input_pattern='*.h5'):

        self.raw_key = raw_key
        self.nuc_seg_key = nuc_seg_key
        self.cell_seg_key = cell_seg_key

        # raw should be 3d, rest should be 2d
        input_ndim = [3, 2, 2]

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.pickle' if identifier is None else f'_{identifier}.pickle'

        super().__init__(input_pattern,
                         output_ext=output_ext,
                         input_key=[self.raw_key,
                                    self.nuc_seg_key,
                                    self.cell_seg_key],
                         input_ndim=input_ndim,
                         identifier=identifier)

    def load_sample(self, path, device):
        with open_file(path, 'r') as f:
            raw = self.read_input(f, self.raw_key)
            nucleus_seg = self.read_input(f, self.nuc_seg_key)
            cell_seg = self.read_input(f, self.cell_seg_key)
        gfp = raw[1]
        serum = raw[2]

        gfp = torch.FloatTensor(gfp.astype(np.float32)).to(device)
        serum = torch.FloatTensor(serum.astype(np.float32)).to(device)
        nucleus_seg = torch.LongTensor(nucleus_seg.astype(np.int32)).to(device)
        cell_seg = torch.LongTensor(cell_seg.astype(np.int32)).to(device)

        cell_seg[nucleus_seg != 0] = 0

        return gfp, serum, nucleus_seg, cell_seg

    def eval_cells(self, gfp, serum, nucleus_seg, cell_seg,
                   ignore_label=0,
                   substract_mean_background=False):
        # all segs have shape H, W
        assert gfp.shape == serum.shape == nucleus_seg.shape == cell_seg.shape
        # include background as instance with label 0
        labels = torch.sort(torch.unique(cell_seg))[0]

        if substract_mean_background:
            gfp -= (gfp[cell_seg == ignore_label]).mean()
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
        cell_properties['gfp'] = get_per_mask_statistics(gfp)
        cell_properties['serum'] = get_per_mask_statistics(serum)
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
            for input_file, output_file in tqdm(list(zip(input_files, output_files))):
                _save_all_stats(input_file, output_file)
