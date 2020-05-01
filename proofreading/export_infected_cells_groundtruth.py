import argparse
import numpy as np
import h5py
import nifty.ground_truth as ngt


def check(raw, seg):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


def export_from_bigcat(in_path, out_path, ignore_label):
    with h5py.File(in_path, 'r') as f:
        raw = f['volumes/raw'][:]
        seg = f['volumes/labels/merged_ids'][0].astype('int64')
        mask = f['volumes/infected_mask'][:]
    assert seg.shape == mask.shape

    seg_ids = np.unique(seg)
    overlaps = ngt.overlap(seg, mask)
    overlaps = {sid: overlaps.overlapArrays(sid, True)[0][0] for sid in seg_ids}

    map_values = {0: 0, 1: 1, 2: ignore_label}
    for seg_id in seg_ids:
        seg[seg == seg_id] = map_values[overlaps[seg_id]]

    assert set(np.unique(seg)) == {0, 1, ignore_label}
    # check(raw, seg)

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
