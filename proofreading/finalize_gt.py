import os
import napari

from batchlib.util import read_image, write_image, write_table, open_file


def get_infected_labels(seg, infected_mask):
    pass


def finalize_gt_volume(data_path, seg_gt_path, infected_gt_path, out_path,
                       plate_name, base_name, check):

    # TODO load all the raw data
    with open_file(data_path, 'r') as f:
        serum_igg = read_image(f, 'serum_IgG')

    shape = serum_igg.shape

    # TODO load the segmentation gt
    with open_file(seg_gt_path, 'r') as f:
        seg = f['cells'][:]
    assert seg.shape == shape

    with open_file(infected_gt_path, 'r') as f:
        infected_mask = f['infected'][:]
    assert infected_mask.shape == shape

    # map infected gt nucleus labels to cell segmentation
    infected_labels_cols, infected_labels_table = get_infected_labels(seg, infected_mask)

    if check:
        inspect()
    else:
        # TODO write everything
        with open_file(out_path, 'a') as f:
            write_image(f, 'serum_IgG', serum_igg)
            write_table(f, 'infected_cell_labels', infected_labels_cols, infected_labels_table)


def inspect():
    pass


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


if __name__ == '__main__':
    gt_root_path = '/home/pape/Work/covid/antibodies-nuclei/groundtruth'
    gt_folder = '/home/pape/Work/covid/antibodies-nuclei/groundtruth/merged'
    data_root = ''
    finalize_all(gt_root_path, gt_folder, data_root)
