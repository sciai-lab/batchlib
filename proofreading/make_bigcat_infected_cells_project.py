import argparse

import h5py
import numpy as np


def fragment_segment_lut(segmentation, infected_labels):
    node_ids = np.unique(segmentation)
    assert len(infected_labels) == len(node_ids)
    assert np.array_equal(np.infected_labels, np.array([0, 1]))

    # the background gets a special label, because it is neither infected nor not infected
    infected_labels[0] = 2
    label_offset = node_ids[-1] + 1

    lut = np.zeros((2, len(node_ids)), dtype='uint64')
    lut[0, :] = node_ids
    lut[1, :] = infected_labels + label_offset

    return lut


def to_uint8(raw):
    raw = raw.astype('float32')
    raw -= raw.min()
    raw /= raw.max()
    raw *= 255
    return raw.astype('uint8')


# TODO need to read Roman's infected / non-infected classification from somewhere.
def to_bigcat(serum, marker, nuclei, segmentation, infected_labels, out_path):
    lut = fragment_segment_lut(segmentation, infected_labels)
    next_id = int(lut.max()) + 1

    res = [1, 1, 1]
    offset = [0, 0, 0]
    with h5py.File(out_path, 'a') as f:
        f.attrs['next_id'] = next_id

        serum = to_uint8(serum)
        marker = to_uint8(marker)
        nuclei = to_uint8(nuclei)

        raw = np.concatenate([serum[None], marker[None], nuclei[None]], axis=0)
        ds = f.create_dataset('volumes/raw', data=raw.astype('uint8'), compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res

        segmentation = segmentation.astype('uint64')
        segmentation = np.concatenate([segmentation[None], np.zeros_like(segmentation)[None]], axis=0)
        ds = f.create_dataset('volumes/labels/fragments', data=segmentation,
                              compression='gzip')
        attrs = ds.attrs
        attrs['resolution'] = res
        attrs['offset'] = offset

        ds = f.require_dataset('fragment_segment_lut', shape=lut.shape, compression='gzip',
                               maxshape=(2, None), dtype='uint64')
        ds[:] = lut


def convert_to_bigcat(in_path, out_path, use_corrected):

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
        seg = f['cell_segmentation/s0'][:]

        # TODO read infected labels from the cell level hdf5 table
        infected_labels = ''

    to_bigcat(serum, marker, nuclei, seg, infected_labels, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--use_corrected', type=int, default=0)
    args = parser.parse_args()

    convert_to_bigcat(args.input_path, args.output_path, args.use_corrected)
