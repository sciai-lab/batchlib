import os
import napari
import nifty.ground_truth as ngt
import nifty.tools as nt
import numpy as np
import vigra

from batchlib.util import read_image, write_image, write_table, open_file


def get_infected_labels_seg_based(seg, infected_mask):
    unique_seg_ids = np.unique(seg)
    overlap = ngt.overlap(seg, infected_mask)

    infected_overlaps = np.array([overlap.overlapArrays(seg_id, True)[0][0]
                                  for seg_id in unique_seg_ids], dtype='uint64')

    new_infected_mask = nt.take(infected_overlaps, seg.astype('uint64'))

    cols = ['label_id', 'infected_label']
    table = np.concatenate([unique_seg_ids[:, None], infected_overlaps[:, None].astype('float32')], axis=1)
    return new_infected_mask, cols, table


def get_infected_labels_nuc_based(seg, infected_mask):
    unique_seg_ids = np.unique(seg)
    overlap = ngt.overlap(seg, infected_mask)

    infected_overlaps = [overlap.overlapArrays(seg_id, True)[0] for seg_id in unique_seg_ids]
    infected_overlaps = [ovlps[1] if (ovlps[0] == 0 and len(ovlps) > 1) else ovlps[0]
                         for ovlps in infected_overlaps]
    infected_overlaps[0] = 0
    infected_overlaps = np.array(infected_overlaps, dtype='uint64')

    new_infected_mask = nt.take(infected_overlaps, seg.astype('uint64'))

    cols = ['label_id', 'infected_label']
    table = np.concatenate([unique_seg_ids[:, None], infected_overlaps[:, None].astype('float32')], axis=1)
    return new_infected_mask, cols, table


def finalize_gt_volume(data_path, seg_gt_path, infected_gt_path, out_path,
                       plate_name, base_name, check):
    assert os.path.exists(data_path), data_path
    assert os.path.exists(seg_gt_path), seg_gt_path
    assert os.path.exists(infected_gt_path), infected_gt_path

    print("Load raw data ...")
    with open_file(data_path, 'r') as f:
        serum_igg = read_image(f, 'serum_IgG_background_subtracted')
        marker = read_image(f, 'marker_background_subtracted')
        nuclei = read_image(f, 'nuclei')

        serum_igg_raw = read_image(f, 'serum_IgG_raw')
        marker_raw = read_image(f, 'marker_raw')
        nuclei_raw = read_image(f, 'nuclei_raw')

        # TODO we should also quickly proofread the nucleus segmentation
        # to get rid of false positives due to dirt
        # also, we should later map nucleus and cell segmentation ids again
        nucleus_seg = read_image(f, 'nucleus_segmentation')

    shape = serum_igg.shape

    # TODO run connected components (excluding bg) and filter small segments
    with open_file(seg_gt_path, 'r') as f:
        seg = f['cells'][:]
        seg = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)[0]
    assert seg.shape == shape

    with open_file(infected_gt_path, 'r') as f:
        infected_mask = f['infected'][:]
    assert infected_mask.shape == shape

    print("Map infected labels ...")
    # TODO infer which one to use properly
    # map infected gt nucleus labels to cell segmentation
    (infected_mask,
     infected_labels_cols,
     infected_labels_table) = get_infected_labels_nuc_based(seg, infected_mask)

    if check:
        print("Inspect volume ...")
        inspect(serum_igg, marker, seg, infected_mask)
    else:
        print("Write all data ...")
        with open_file(out_path, 'a') as f:
            write_image(f, 'serum_IgG', serum_igg)
            write_image(f, 'marker', marker)
            write_image(f, 'nuclei', nuclei)

            write_image(f, 'serum_IgG', serum_igg_raw)
            write_image(f, 'marker_raw', marker_raw)
            write_image(f, 'nuclei_raw', nuclei_raw)

            write_image(f, 'cell_segmentation', seg)
            write_image(f, 'nucleus_segmentation', nucleus_seg)
            write_image(f, 'infected_mask', infected_mask)

            write_table(f, 'infected_cell_labels', infected_labels_cols, infected_labels_table)

            f.attrs['plate'] = plate_name
            f.attrs['image'] = base_name


def inspect(serum_igg, marker, seg, infected_mask):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(serum_igg, name='serum')
        viewer.add_image(marker, name='marker')
        viewer.add_labels(seg, name='segmentation')
        viewer.add_labels(infected_mask, name='infected-mask')


def finalize_all(gt_root_path, gt_folder, data_root, check=False):
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
            data_path = os.path.join(data_root, base_name + '.h5')

            out_path = os.path.join(gt_folder, 'gt_image_%03i.h5' % ii)
            finalize_gt_volume(data_path, seg_gt_path, infected_gt_path,
                               out_path, plate_name, base_name, check=check)
            # print(infected_gt_path)
            # print(seg_gt_path)

            ii += 1


def test_single_gt():
    seg_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/segmentation/20200420_164920_764_IgG/WellD11_PointD11_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5,DIA_Seq0270_segmentation_done.h5'
    infected_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/infection/20200420_164920_764_IgG/WellD11_PointD11_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5,DIA_Seq0270_infected.h5'
    data_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/data/WellD11_PointD11_0000_ChannelDAPI,WF_GFP,TRITC,WF_Cy5,DIA_Seq0270.h5'

    out_path = './test.h5'
    base_name = os.path.splitext(os.path.split(data_path)[1])[0]
    plate_name = '20200420_164920_764'

    finalize_gt_volume(data_path, seg_path, infected_path, out_path,
                       plate_name, base_name, False)


def finalize_initial():
    root = '/home/pape/Work/covid/antibodies-nuclei/groundtruth'
    data_root = os.path.join(root, 'data/20200405_test_images')
    seg_root = os.path.join(root, 'segmentation/20200405_test_images')
    infected_root = os.path.join(root, 'infection/20200405_test_images')

    for im_name in os.listdir(data_root):
        print(im_name)

        name = os.path.splitext(im_name)[0]
        finalize_gt_volume(os.path.join(data_root, im_name),
                           os.path.join(seg_root, name + '_segmentation_done.h5'),
                           os.path.join(infected_root, name + '_infected_nuclei.h5'),
                           out_path=name + '.h5',
                           plate_name='20200405_test_images',
                           base_name=name,
                           check=False)


if __name__ == '__main__':
    # test_single_gt()

    finalize_initial()

    # gt_root_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth'
    # gt_folder = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/merged'
    # data_root = ''
    # finalize_all(gt_root_path, gt_folder, data_root)
