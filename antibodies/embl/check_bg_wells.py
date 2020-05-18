import os
import json
import h5py
from batchlib.util import read_table


def has_two_bg_wells(name):
    isT = 'plateT' in name

    is_recentK = False
    if 'plateK' in name or 'PlateK' in name:
        prelen = len('plateK')
        kid = int(name[prelen:prelen+2])
        if kid >= 22:
            is_recentK = True

    return isT or is_recentK


def get_expected_bg_well(name, bg_wells_per_plate):
    if name in bg_wells_per_plate:
        return bg_wells_per_plate[name]
    if has_two_bg_wells(name):
        return ['H01', 'G01']
    else:
        return ['H01']


def get_channel_dict(name):
    root_in = '/g/kreshuk/data/covid/covid-data-vibor'
    channel_mapping = os.path.join(root_in, name, 'channel_mapping.json')
    with open(channel_mapping) as f:
        channel_mapping = json.load(f)
    channel_dict = {name: f'{name}_min_well' for name in channel_mapping.values()
                    if name is not None and name.startswith('serum')}
    return channel_dict


def get_actual_bg_well(name, root, channel_dict, bg_table_key='plate/backgrounds_from_min_well'):
    folder = os.path.join(root, name)
    table_path = os.path.join(folder, f'{name}_table.hdf5')
    with h5py.File(table_path, 'r') as f:
        col_names, table = read_table(f, bg_table_key)
    return {chan_name: table[0, col_names.index(well_key)] for chan_name, well_key in channel_dict.items()}


def check_bg_well_for_all_plates(root):
    plate_names = os.listdir(root)
    plate_names.sort()

    bg_wells_per_plate = '../../misc/plates_to_background_well.json'
    with open(bg_wells_per_plate) as f:
        bg_wells_per_plate = json.load(f)

    for name in plate_names:
        expected_bg_well = get_expected_bg_well(name, bg_wells_per_plate)
        channel_dict = get_channel_dict(name)
        actual_bg_well = get_actual_bg_well(name, root, channel_dict)
        print(name)
        print(expected_bg_well)
        print(actual_bg_well)
        print()


if __name__ == '__main__':
    root = '/g/kreshuk/data/covid/data-processed-scratch'
    check_bg_well_for_all_plates(root)
