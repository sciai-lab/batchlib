
import h5py
import numpy as np

from glob import glob


# This is a one time script.
# Therefore I will not bother with command-line arguments

# Arguments
input_pattern = "/home/covid19/gt-for-steffen/merged_gt/*.h5"



for gt_file in glob(input_pattern):

    with h5py.File(gt_file, mode="r+") as inp:
        nucleus_segmentation = inp["nucleus_segmentation/s0"][:]
        infected = inp["infected_mask/s0"][:].astype(np.int64)

        gt_seg = np.zeros_like(nucleus_segmentation, dtype=np.int64)

        for idx in np.unique(nucleus_segmentation):
            
            # skip background
            if idx == 0:
                continue

            # find most common infected/non-infected label in infected array
            # we expect 0 = ignore, 1 = infected, 2 = not infected
            nucleus_mask = (nucleus_segmentation == idx)
            counts = np.bincount(infected[nucleus_mask], minlength=3)

            # check if there is at least one pixel without ignore label
            if np.sum(counts[1:]) > 0:

                if counts[1] == counts[2]:
                    # same number of infected and not infected pixels in nucleus
                    # set it to ignore label == -1
                    gt_seg[nucleus_mask] = -1
                    print(f"Warning: same number of infected and not infected ")
                    print(f"pixels found in cell {idx} in {gt_file}")
                    continue

                cell_label = 1 if counts[1] > counts[2] else 2
                gt_seg[nucleus_mask] = cell_label
            else:
                print(f"Warning: only ignore label for cell {idx} in {gt_file}")

        inp.create_dataset("infected_nuclei/s0", data=gt_seg, compression="gzip")