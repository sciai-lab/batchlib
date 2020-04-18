import os
import json
import re
from glob import glob
from tqdm import tqdm
from copy import copy


def _print_dict(d):
    [print(f'{key} -> {value}') for key, value in d.items()]


def split_by_serum_channel(data_dir, channel_mapping_file='channel_mapping.json'):
    print(f'Making directories with individual serum channels for {data_dir}.')
    data_dir = os.path.expanduser(data_dir)
    # load channel mapping with multiple serum channels
    with open(os.path.join(data_dir, channel_mapping_file), 'r') as f:
        channel_mapping_multi = json.load(f)
    print(f'Found channel mapping:')
    _print_dict(channel_mapping_multi)

    serum_identifiers = []
    for value in channel_mapping_multi.values():
        if isinstance(value, str) and value.startswith('serum_'):
            serum_identifiers.append(value[6:])

    data_base_dir = os.path.dirname(data_dir)
    plate_dir = os.path.basename(data_dir)

    for serum_identifier in serum_identifiers:
        new_plate_dir = plate_dir + '_' + serum_identifier
        if os.path.exists(os.path.join(data_base_dir, new_plate_dir)):
            print(f'Skipping serum identifier {serum_identifier} because directory {new_plate_dir} exists already')
            continue
        print(f'Making plate dir {new_plate_dir} for serum_{serum_identifier}')
        os.makedirs(os.path.join(data_base_dir, new_plate_dir))

        # symlink tiff files
        for f in tqdm(list(glob(os.path.join(data_dir, '*.tiff'))), desc='symlinking .tiff files'):
            os.symlink(f, f.replace(plate_dir, new_plate_dir))

        channel_mapping = copy(channel_mapping_multi)
        for key, value in channel_mapping.items():
            if value in ['nuclei', 'marker'] or value is None:
                continue
            elif value == 'serum_' + serum_identifier:
                channel_mapping[key] = 'serum'
            else:
                channel_mapping[key] = None
        print('new channel mapping:')
        _print_dict(channel_mapping)
        with open(os.path.join(data_base_dir, new_plate_dir, 'channel_mapping.json'), 'w') as f:
            json.dump(channel_mapping, f)
    print('Done.\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dirs', nargs='+', type=str, help='directory to process')
    args = parser.parse_args()
    for d in args.data_dirs:
        if d.endswith(os.path.sep):
            d = d[:-1]
        split_by_serum_channel(data_dir=d)
