import os
import json
from glob import glob


def update_channel_mapping(folder):
    mapping_file = os.path.join(folder, 'channel_mapping.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    values = list(mapping.values())
    if 'serum' in values:
        print(folder, "needs update")


def update_channel_mappings(root):
    folders = glob(os.path.join(root, '*'))
    for folder in folders:
        update_channel_mapping(folder)


if __name__ == '__main__':
    root = ''
    update_channel_mappings(root)
