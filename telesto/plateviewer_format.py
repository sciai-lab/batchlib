import argparse
import os

import pandas as pd


def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)


def next_name(well_name, well_id, image_id, to_next):
    if to_next and well_id == 12:
        well_name = next_alpha(well_name)
        well_id = 1
        image_id = 0
    elif to_next:
        well_id += 1
        image_id = 0
    else:
        image_id += 1
    return well_name, well_id, image_id


def to_output(folder, well_name, well_id, image_id):
    well = '%s%02i' % (well_name, well_id)
    name = 'Well%s_Point%s_%04i_ChannelIgG,IgA,DAPI_Seq' % (well, well, image_id)
    path = os.path.join(folder, f'{name}.h5')
    return path


def to_plate_viewer_format(input_folder, output_folder, table_path):
    os.makedirs(output_folder, exist_ok=True)
    table = pd.read_csv(table_path, sep='\t')

    names = table['id'].values
    groups = table['group'].values

    last_group = groups[0]

    well_name = 'A'
    well_id = 1
    image_id = -1

    for name, group in zip(names, groups):
        input_path = os.path.join(input_folder, f'{name}.h5')
        well_name, well_id, image_id = next_name(well_name, well_id, image_id, group != last_group)
        output_path = to_output(output_folder, well_name, well_id, image_id)
        last_group = group
        os.symlink(input_path, output_path)

    # TODO link the hdf5 table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')

    args = parser.parse_args()
    input_folder = args.input_folder
    name = os.path.split(input_folder)[1]
    table_path = os.path.join(input_folder, f'{name}.tsv')

    input_folder = input_folder.replace('telesto', 'telesto/data-processed')
    output_folder = input_folder + '_plateviewer'
    to_plate_viewer_format(input_folder, output_folder, table_path)
