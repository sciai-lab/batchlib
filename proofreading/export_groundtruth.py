import argparse
import numpy as np
import h5py


# TODO enable exporting more channels for the infected problem ?
def export_from_bigcat(in_path, out_path):
    with h5py.File(in_path, 'r') as f:
        raw = f['volumes/raw'][0, :, :]
        seg = f['volumes/labels/merged_ids'][0, :, :]

    # largest segment is background
    ids, sizes = np.unique(seg, return_counts=True)
    size_sorted = np.argsort(sizes)[::-1]
    bg_id = ids[size_sorted[0]]

    seg[seg == bg_id] = 0

    with h5py.File(out_path, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('cells', data=seg, compression='gzip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    export_from_bigcat(args.input_path, args.output_path)
