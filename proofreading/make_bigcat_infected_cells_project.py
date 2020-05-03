import argparse

import h5py
import numpy as np


def fragment_segment_lut(segmentation, infected_labels):
    node_ids = np.unique(segmentation)
    assert len(infected_labels) == len(node_ids)
    assert np.array_equal(np.unique(infected_labels), np.array([0, 1]))
    infected_labels += 1

    # the background gets a special label, because it is neither infected nor not infected
    infected_labels[0] = 0
    label_offset = node_ids[-1] + 1

    lut = np.zeros((2, len(node_ids)), dtype='uint64')
    lut[0, :] = node_ids
    lut[1, :] = infected_labels + label_offset

    return lut


def to_uint8(raw):
    raw = raw.astype('float32')
    raw_min = raw.min()
    raw -= raw_min
    raw /= raw.max() - raw_min
    raw *= 255
    return raw.astype('uint8')


def make_infected_mask(segmentation, labels):
    infected_mask = np.isin(segmentation, np.where(labels == 1)[0]).astype('int8')
    infected_mask[infected_mask == 1] = 2
    infected_mask[infected_mask == 0] = 1
    infected_mask[segmentation == 0] = 0
    return infected_mask


def to_bigcat(serum, marker, nuclei, segmentation, infected_labels, out_path, order):
    lut = fragment_segment_lut(segmentation, infected_labels)
    next_id = int(lut.max()) + 1

    # we need to save the initial infected mask
    # so we can map back the ids later
    infected_mask = make_infected_mask(segmentation, infected_labels)

    res = [1, 1, 1]
    offset = [0, 0, 0]
    with h5py.File(out_path, 'a') as f:
        f.attrs['next_id'] = next_id

        serum = to_uint8(serum)
        marker = to_uint8(marker)
        nuclei = to_uint8(nuclei)

        raw = {"s": serum[None], "m": marker[None], "n": nuclei[None]}
        raw = np.concatenate([raw[order[0]], raw[order[1]], raw[order[2]]], axis=0)
        ds = f.create_dataset('volumes/raw', data=raw.astype('uint8'), compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res

        segmentation = segmentation.astype('uint64')
        segmentation = segmentation[None]
        ds = f.create_dataset('volumes/labels/fragments', data=segmentation, compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res
        attrs['offset'] = offset

        ds = f.require_dataset('fragment_segment_lut', shape=lut.shape, compression='gzip',
                               maxshape=(2, None), dtype='uint64')
        ds[:] = lut

        ds = f.create_dataset('volumes/infected_mask', data=infected_mask, compression='gzip')


def infected_labels_from_table(f, infected_label_key):
    g = f[infected_label_key]
    table = g['cells'][:]
    cols = [name.decode('utf8') for name in g['columns'][:]]
    idx = cols.index('is_infected')
    return table[:, idx].astype('float').astype('uint8')


def infected_labels_from_mask(f, infected_mask_key, seg):
    import nifty.ground_truth as ngt
    ds = f[infected_mask_key]['s0']
    mask = ds[:].astype('uint32')

    node_ids = np.unique(seg)
    overlap = ngt.overlap(seg, mask)

    infected_labels = [overlap.overlapArrays(nid, True)[0][0] for nid in node_ids]
    infected_labels = np.array(infected_labels, dtype='uint8')

    return infected_labels


# TODO get keys via argparse too
def convert_to_bigcat(in_path, out_path, use_corrected, order,
                      infected_label_key='tables/cell_classification/cell_segmentation/marker_corrected',
                      infected_mask_key='infected_cell_mask'):

    if use_corrected:
        serum_key = 'serum_corrected'
        marker_key = 'marker_corrected'
        nuclei_key = 'nuclei_corrected'
    else:
        serum_key = 'serum'
        marker_key = 'marker'
        nuclei_key = 'nuclei_corrected'

    with h5py.File(in_path, 'r') as f:
        serum = f[serum_key]['s0'][:]
        marker = f[marker_key]['s0'][:]
        nuclei = f[nuclei_key]['s0'][:]
        seg = f['nucleus_segmentation/s0'][:]

        if infected_label_key in f:
            print(f"Reading infected labels from table @{infected_label_key}")
            infected_labels = infected_labels_from_table(f, infected_label_key)
        elif infected_mask_key in f:
            print(f"Reading infected labels from mask @{infected_mask_key}")
            infected_labels = infected_labels_from_mask(f, infected_mask_key, seg)
        else:
            raise ValueError(f"Could neither find the table @{infected_label_key} or the mask @{infected_mask_key}")

    to_bigcat(serum, marker, nuclei, seg, infected_labels, out_path, order)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--order', type=str, default="mns",
                        help="example oder: smn (serum, marker, nuclei) or mns")
    parser.add_argument('--use_corrected', type=int, default=0)
    args = parser.parse_args()

    convert_to_bigcat(args.input_path, args.output_path, args.use_corrected, args.order)
