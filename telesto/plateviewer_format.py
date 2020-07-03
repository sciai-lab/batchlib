import argparse
import os

import h5py
from batchlib.util import read_table, write_table
from batchlib.workflows.telesto import export_image_table


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


def to_names(well_name, well_id, image_id):
    well = '%s%02i' % (well_name, well_id)
    name = 'Well%s_Point%s_%04i_ChannelIgG,IgA,DAPI_Seq' % (well, well, image_id)
    site_name = '%s_%04i' % (well, image_id)
    return name, site_name


def check_next_group(have_group_info, group, last_group, image_id, group_size):
    if have_group_info:
        return group != last_group
    else:
        return image_id % (group_size - 1) == 0


def to_plate_viewer_format(input_folder, output_folder, group_size=9):
    os.makedirs(output_folder, exist_ok=True)

    plate_name = os.path.split(input_folder)[1]
    input_table_path = os.path.join(input_folder, f'{plate_name}_table.hdf5')

    with h5py.File(input_table_path, 'r') as f:
        column_names, table = read_table(f, 'images/default')

    names = table[:, column_names.index('image_name')]
    groups = table[:, column_names.index('group_name')]

    have_group_info = not (groups == '').all()

    last_group = groups[0]

    well_name = 'A'
    well_id = 1
    image_id = -1

    new_table = table.copy()
    new_column_names = column_names
    new_column_names[column_names.index('group_name')] = 'site_name'

    for ii, (name, group) in enumerate(zip(names, groups)):
        input_path = os.path.join(input_folder, f'{name}.h5')

        is_next_group = check_next_group(have_group_info, group, last_group, image_id, group_size)

        well_name, well_id, image_id = next_name(well_name, well_id, image_id, is_next_group)
        image_name, site_name = to_names(well_name, well_id, image_id)

        output_path = os.path.join(output_folder, image_name)
        last_group = group
        os.symlink(input_path, output_path)

        new_table[:, new_column_names.index('image_name')] = image_name
        new_table[:, new_column_names.index('site_name')] = site_name

    # write the new hdf5 table
    out_table_path = os.path.join(output_folder, f'{plate_name}_table.hdf5')
    with h5py.File(out_table_path, 'a') as f:
        write_table(f, 'images/default', new_column_names, new_table)

    # write the new excel table
    export_image_table(output_folder, plate_name=plate_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')

    args = parser.parse_args()
    input_folder = args.input_folder.rstrip('/')
    output_folder = input_folder + '_plateviewer'

    to_plate_viewer_format(input_folder, output_folder)
