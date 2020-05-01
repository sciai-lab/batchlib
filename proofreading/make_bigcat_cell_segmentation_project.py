import argparse

import h5py
import vigra
import nifty.ground_truth as ngt
import numpy as np


def fragment_segment_lut(watershed, segmentation):
    overlap = ngt.overlap(watershed, segmentation)
    node_ids = np.unique(watershed)

    # TODO is this sorted in correct order ?
    assignments = [overlap.overlapArrays(nid, True)[0][0] for nid in node_ids]
    assignments = np.array(assignments, dtype='uint64')

    vigra.analysis.relabelConsecutive(assignments, start_label=1, keep_zeros=False, out=assignments)
    max_node_id = node_ids[-1]

    lut = np.zeros((2, len(assignments)), dtype='uint64')
    lut[0, :] = node_ids
    lut[1, :] = assignments + max_node_id

    return lut


def to_uint8(raw):
    raw = raw.astype('float32')
    raw -= raw.min()
    raw /= raw.max()
    raw *= 255
    return raw.astype('uint8')


def to_bigcat(serum, nuclei, watershed, segmentation, out_path):
    lut = fragment_segment_lut(watershed, segmentation)
    next_id = int(lut.max()) + 1

    res = [1, 1, 1]
    offset = [0, 0, 0]
    with h5py.File(out_path, 'a') as f:
        f.attrs['next_id'] = next_id

        serum = to_uint8(serum)
        nuclei = to_uint8(nuclei)
        raw = np.concatenate([serum[None], nuclei[None]], axis=0)
        ds = f.create_dataset('volumes/raw', data=raw.astype('uint8'), compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res

        watershed = watershed[None].astype('uint64')
        ds = f.create_dataset('volumes/labels/fragments', data=watershed, compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res
        attrs['offset'] = offset

        ds = f.require_dataset('fragment_segment_lut', shape=lut.shape, compression='gzip',
                               maxshape=(2, None), dtype='uint64')
        ds[:] = lut


def compute_oversegmentation(raw):
    sigma = 2.
    edges = vigra.filters.gaussianGradientMagnitude(raw.astype('float32'), sigma)
    edges = vigra.filters.gaussianSmoothing(edges, sigma)

    seeds = vigra.analysis.localMinima(edges, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    seeds = np.isnan(seeds).astype('uint32')
    seeds = vigra.analysis.labelImageWithBackground(seeds)

    ws, _ = vigra.analysis.watershedsNew(edges, seeds=seeds)

    return ws


def convert_to_bigcat(in_path, out_path, use_corrected):

    if use_corrected:
        serum_key = 'serum_corrected'
        nuclei_key = 'nuclei_corrected'
    else:
        serum_key = 'serum'
        nuclei_key = 'nuclei'

    with h5py.File(in_path, 'r') as f:
        serum = f[serum_key]['s0'][:]
        nuclei = f[nuclei_key]['s0'][:]
        seg = f['cell_segmentation/s0'][:]

    ws = compute_oversegmentation(serum)
    to_bigcat(serum, nuclei, ws, seg, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--use_corrected', type=int, default=0)
    args = parser.parse_args()

    convert_to_bigcat(args.input_path, args.output_path, args.use_corrected)
