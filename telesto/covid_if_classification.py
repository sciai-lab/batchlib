import os
from glob import glob
import numpy as np
import pandas as pd

MANUSCRIPT_PLATES = [
    "20200417_132123_311",
    "20200417_152052_943",
    "20200417_172611_193",
    "20200417_185943_790",
    "20200420_152417_316",
    "20200420_164920_764",
    "plate1_IgM_20200527_125952_707",
    "plate2_IgM_20200527_155923_897",
    "plate5_IgM_20200528_094947_410",
    "plate6_IgM_20200528_111507_585",
    "plate7_IgM_20200602_162805_201",
    "plate7rep1_20200426_103425_693",
    "plate8_IgM_20200529_144442_538",
    "plate8rep1_20200425_162127_242",
    "plate8rep2_20200502_182438_996",
    "plate9_2rep2_20200515_230124_149",
    "plate9_3rep1_20200513_160853_327",
    "plate9_4_IgM_20200604_212451_328",
    "plate9_4rep1_20200604_175423_514",
    "plate9_5_IgM_20200605_084742_832",
    "plate9_5rep1_20200604_225512_896",
    "plate9rep1_20200430_144438_974"
]


def get_label(label_dict, plate_name, im_path):
    im_name = os.path.splitext(os.path.split(im_path)[1])[0]
    well_name = im_name.split('_')[0][4:]
    label = label_dict[plate_name][well_name]
    if label == 'positive':
        label = 1
    elif label == 'control':
        label = 0
    else:
        label = None
    return label


def to_label_dict(res_table_path):
    res_table = pd.read_excel(res_table_path)
    plate_names = res_table['plate_name']
    well_names = res_table['well_name']
    labels = res_table['cohort_type']

    unique_plate_names = np.unique(plate_names)

    label_dict = {}
    for plate_name in unique_plate_names:
        plate_mask = plate_names == plate_name
        label_dict[plate_name] = dict(zip(well_names[plate_mask], labels[plate_mask]))

    return label_dict


def to_classification_task(input_root, plate_names, res_table_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    label_dict = to_label_dict(res_table_path)

    id_col = []
    label_col = []

    counter = 0
    for plate in plate_names:
        print("Processing plate", plate)
        plate_folder = os.path.join(input_root, plate)
        assert os.path.exists(plate_folder), plate_folder

        in_files = glob(os.path.join(plate_folder, '*.h5'))
        for in_file in in_files:
            label = get_label(label_dict, plate, in_file)
            if label is None:
                print("Skipping", in_file)
                continue

            label_col.append(label)

            im_name = 'im_%06i' % counter
            out_path = os.path.join(out_folder, im_name + '.h5')
            counter += 1

            os.symlink(in_file, out_path)
            id_col.append(im_name)

    n_images = len(id_col)
    assert len(label_col) == n_images

    id_col = np.array(id_col)
    label_col = np.array(label_col)

    n_pos = (label_col == 1).sum()
    n_neg = (label_col == 0).sum()
    assert n_pos + n_neg == n_images
    print("Number of images:", n_images)
    print("Number of negative:", n_neg)
    print("Number of positives:", n_pos)

    columns = ['id', 'label_i']
    table = np.concatenate([id_col[:, None], label_col[:, None]], axis=1)

    out_name = os.path.split(output_folder)[1]
    table_out_path = os.path.join(output_folder, f'{out_name}.tsv')
    table = pd.DataFrame(table, columns=columns)
    table.to_csv(table_out_path, columns=columns, sep='\t', index=False)


if __name__ == '__main__':
    root = '/g/kreshuk/data/covid/data-processed'
    out_folder = '/g/kreshuk/data/covid/telesto/for_pretraining'
    res_table = '/g/kreshuk/data/covid/manuscript_plates_20200611_scores.xlsx'
    to_classification_task(root, MANUSCRIPT_PLATES, res_table, out_folder)
