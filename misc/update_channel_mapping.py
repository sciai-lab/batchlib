import os
import json
from glob import glob


def update_channel_mapping(folder, plate_name):
    mapping_file = os.path.join(folder, 'channel_mapping.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    names = list(mapping.values())
    serum_names = [name for name in names if name is not None and name.startswith('serum')]

    print(plate_name, ':', serum_names)


def update_channel_mappings(root):
    folders = glob(os.path.join(root, '*'))
    folders.sort()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        name = os.path.split(folder)[1]
        if name == 'deprecated' or name == 'tiny_test':
            continue
        update_channel_mapping(folder, name)


if __name__ == '__main__':
    root = '/g/kreshuk/data/covid/covid-data-vibor'
    update_channel_mappings(root)
