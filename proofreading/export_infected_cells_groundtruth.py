import argparse
import numpy as np
import h5py
import nifty.ground_truth as ngt


def check(raw, seg, mask):
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)
        viewer.add_labels(mask)


def export_from_bigcat(in_path, out_path):
    with h5py.File(in_path, 'r') as f:
        raw = f['volumes/raw'][:]
        seg = f['volumes/labels/merged_ids'][0].astype('int64')
        mask = f['volumes/infected_mask'][:]
    assert seg.shape == mask.shape

    seg_ids = np.unique(seg)
    overlaps = ngt.overlap(seg, mask)
    overlaps = {sid: overlaps.overlapArrays(sid, True)[0][0] for sid in seg_ids}

    for seg_id in seg_ids:
        seg[seg == seg_id] = overlaps[seg_id]

    assert set(np.unique(seg)) == {0, 1, 2}
    # check(raw, seg, mask)

    with h5py.File(out_path, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('infected', data=seg, compression='gzip')


def export_all():
    import os
    from glob import glob
    out_folder = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/infection/20200405_test_images'
    names = glob('./old_gt/*.h5')
    names.sort()
    for ii, name in enumerate(names, 1):
        name = os.path.splitext(os.path.split(name)[1])[0]
        out_path = os.path.join(out_folder, name + '_infected_nuclei.h5')
        in_path = './bkp%i.h5' % ii
        print(in_path)
        print(out_path)
        print()
        export_from_bigcat(in_path, out_path)


if __name__ == '__main__':
    # export_all()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    export_from_bigcat(args.input_path, args.output_path)
