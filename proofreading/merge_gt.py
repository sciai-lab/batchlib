import os
from glob import glob

import napari
import nifty.ground_truth as ngt
import nifty.tools as nt
import numpy as np
import vigra

from batchlib.util import read_image, write_image, write_table, open_file

DEFAULT_RAW_KEYS = ('serum_IgG', 'nuclei', 'marker')


def get_infected_labels(seg, infected_mask, invert=False):
    unique_seg_ids = np.unique(seg)
    overlap = ngt.overlap(seg, infected_mask)

    infected_overlaps = [overlap.overlapArrays(seg_id, True)[0] for seg_id in unique_seg_ids]
    infected_overlaps = [ovlps[1] if (ovlps[0] == 0 and len(ovlps) > 1) else ovlps[0]
                         for ovlps in infected_overlaps]
    infected_overlaps[0] = 0
    if invert:
        inv_dict = {0: 0, 1: 2, 2: 1}
        infected_overlaps = [inv_dict[ovlp] for ovlp in infected_overlaps]

    infected_overlaps = np.array(infected_overlaps, dtype='uint64')
    assert len(infected_overlaps) == len(unique_seg_ids)

    new_infected_mask = nt.take(infected_overlaps, seg.astype('uint64'))

    cols = ['label_id', 'infected_label']
    table = np.concatenate([unique_seg_ids[:, None], infected_overlaps[:, None].astype('float32')], axis=1)
    return new_infected_mask, cols, table


def size_filter(seg, min_size=100):
    seg = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)[0]
    seg_ids, counts = np.unique(seg, return_counts=True)
    filter_ = counts < min_size
    filter_ids = seg_ids[filter_]

    bg = seg == 0
    seg[np.isin(seg, filter_ids)] = 0
    hmap = np.random.rand(*seg.shape).astype('float32')
    seg, _ = vigra.analysis.watershedsNew(hmap, seeds=seg.astype('uint32'))
    seg[bg] = 0

    seg = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)[0]
    return seg


def merge_gt_volume(data_path, seg_gt_path, infected_gt_path,
                    out_path, plate_name, base_name,
                    check=False, raw_keys=DEFAULT_RAW_KEYS):
    assert os.path.exists(data_path), data_path
    assert os.path.exists(seg_gt_path), seg_gt_path
    assert os.path.exists(infected_gt_path), infected_gt_path

    print("Load raw data ...")
    raw_dict = {}
    shape = None
    with open_file(data_path, 'r') as f:
        for key in raw_keys:
            data = read_image(f, key)
            raw_dict[key] = data
            if shape is None:
                shape = data.shape

        nucleus_seg = read_image(f, 'nucleus_segmentation')

    with open_file(seg_gt_path, 'r') as f:
        seg = f['cells'][:]
    seg = size_filter(seg)
    assert seg.shape == shape

    with open_file(infected_gt_path, 'r') as f:
        infected_mask = f['infected'][:]
    assert infected_mask.shape == shape

    print("Map infected labels ...")
    invert_infected = (plate_name == "20200417_172611_193_IgG" and
                       base_name == "WellC06_PointC06_0003_ChannelDAPI,WF_GFP,TRITC,WF_Cy5,DIA_Seq0201")
    (new_infected_mask,
     infected_labels_cols,
     infected_labels_table) = get_infected_labels(seg, infected_mask,
                                                  invert=invert_infected)

    if check:
        print("Inspect volume ...")
        inspect(raw_dict, seg, infected_mask, new_infected_mask, plate_name, base_name)
    else:
        print("Write all data ...")
        with open_file(out_path, 'a') as f:
            for name, raw in raw_dict.items():
                write_image(f, name, raw)

            write_image(f, 'cell_segmentation', seg)
            write_image(f, 'nucleus_segmentation', nucleus_seg)
            write_image(f, 'infected_mask', new_infected_mask)

            write_table(f, 'infected_cell_labels', infected_labels_cols, infected_labels_table)

            f.attrs['plate'] = plate_name
            f.attrs['image'] = base_name


def inspect(raw_dict, seg, infected_mask, new_infected_mask, plate_name, base_name):
    with napari.gui_qt():
        viewer = napari.Viewer(title=f'{plate_name}/{base_name}')
        for name, raw in raw_dict.items():
            viewer.add_image(raw, name=name)
        viewer.add_labels(seg, name='segmentation')
        viewer.add_labels(infected_mask, name='infected-mask')
        viewer.add_labels(new_infected_mask, name='new-infected-mask')


def merge_all(gt_root_path, gt_folder, data_root, check=False):
    os.makedirs(gt_folder, exist_ok=True)

    infection_root = os.path.join(gt_root_path, 'infection')
    seg_root = os.path.join(gt_root_path, 'segmentation')

    ii = 0
    for root, dirs, files in os.walk(infection_root):
        for ff in files:
            if not ff.endswith('nuclei.h5'):
                continue
            base_name = ff.rstrip('infected_nuclei.h5')
            infected_gt_path = os.path.join(root, ff)
            plate_name = os.path.relpath(root, infection_root)
            seg_gt_path = os.path.join(seg_root, plate_name, base_name + '_segmentation_done.h5')
            if not os.path.exists(seg_gt_path):
                continue
            data_path = os.path.join(data_root, plate_name, base_name + '.h5')

            out_path = os.path.join(gt_folder, 'gt_image_%03i.h5' % ii)
            merge_gt_volume(data_path, seg_gt_path, infected_gt_path,
                            out_path, plate_name, base_name, check=check)
            ii += 1


def check_files(gt_root_path, skip_mising=True):

    def _check(ref, test, pattern1, pattern2):
        folder_names = os.listdir(ref)

        for folder in folder_names:
            folder_ref = os.path.join(ref, folder)
            folder_test = os.path.join(test, folder)

            have_test = os.path.exists(folder_test)
            if not have_test:
                msg = f'{folder_test} is missing'
                if skip_mising:
                    print(msg)
                    continue
                else:
                    raise RuntimeError(msg)

            file_names = glob(os.path.join(folder_ref, f'*{pattern1}.h5'))
            file_names = [os.path.split(fname)[1] for fname in file_names]
            for fname in file_names:
                f_test = os.path.join(folder_test, fname.replace(pattern1, pattern2))
                if not os.path.exists(f_test):
                    msg = f'{f_test} is missing'
                    if skip_mising:
                        print(msg)
                        continue
                    else:
                        raise RuntimeError(msg)

    pattern_infected = 'infected_nuclei'
    pattern_segmentation = 'segmentation_done'
    _check(os.path.join(gt_root_path, 'infection'),
           os.path.join(gt_root_path, 'segmentation'),
           pattern_infected, pattern_segmentation)
    print()
    _check(os.path.join(gt_root_path, 'segmentation'),
           os.path.join(gt_root_path, 'infection'),
           pattern_segmentation, pattern_infected)


if __name__ == '__main__':
    gt_root_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth'
    data_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/data'
    out_folder = 'merged_gt'
    check = False

    # TODO make sure we have everything and then set skip missing to false
    check_files(gt_root_path, skip_mising=True)
    merge_all(gt_root_path, out_folder, data_path, check=check)
