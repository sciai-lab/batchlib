import argparse
import h5py


def export_from_bigcat(in_path, out_path, ignore_label):
    with h5py.File(in_path, 'r') as f:
        raw = f['volumes/raw'][:]
        seg = f['volumes/labels/merged_ids'][0, :, :].astype('int64')

    # need to bring labels back to [0, 1, 2]
    # 0 - not infected label
    # 1 - infected label
    # 2 - background label
    seg -= seg.min()

    # remap the ignore label
    seg[seg == 2] = ignore_label

    with h5py.File(out_path, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('infected', data=seg, compression='gzip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--ignore_label', type=int, default=-1)
    args = parser.parse_args()

    export_from_bigcat(args.input_path, args.output_path, args.ignore_label)
