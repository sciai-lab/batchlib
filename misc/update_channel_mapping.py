import os
import json
from datetime import datetime
from glob import glob

NEW_SERUM_CHANNES = {
 '20200406_164555_328': ['serum_IgG'],
 '20200406_210102_953': ['serum_IgG'],
 '20200406_222205_911': ['serum_IgG'],
 '20200410_145132_254': ['serum_IgG'],
 '20200410_172801_461': ['serum_IgA'],
 # NOTE this case actually contains both IgG and IgA:
 # 20200415_150710_683 : ['serum_IgA'] column 1-6 ['serum_IgG'] column 7-12
 # for now we will ignore it and just label it as IgG
 '20200415_150710_683': ['serum_IgG'],
 '20200417_132123_311': ['serum_IgG', 'serum_IgA'],
 '20200417_152052_943': ['serum_IgG', 'serum_IgA'],
 '20200417_172611_193': ['serum_IgG', 'serum_IgA'],
 '20200417_185943_790': ['serum_IgG', 'serum_IgA'],
 '20200417_203228_156': ['serum_IgG', 'serum_IgA'],
 '20200420_152417_316': ['serum_IgG', 'serum_IgA'],
 '20200420_164920_764': ['serum_IgG', 'serum_IgA'],
 'plate7rep1_20200426_103425_693': ['serum_IgG', 'serum_IgA'],
 'plate8rep1_20200425_162127_242': ['serum_IgG', 'serum_IgA'],
 'plate9rep1_20200430_144438_974': ['serum_IgG', 'serum_IgA'],
 'plateK10rep1_20200429_122048_065': ['serum_IgG', 'serum_IgA'],
 'plateK11rep1_20200429_140316_208': ['serum_IgG', 'serum_IgA'],
 'plateK12rep1_20200430_155932_313': ['serum_IgG', 'serum_IgA'],
 'plateK13rep1_20200430_175056_461': ['serum_IgG', 'serum_IgA'],
 'plateK14rep1_20200430_194338_941': ['serum_IgG', 'serum_IgA'],
 'titration_plate_20200403_154849': ['serum_IgG']
}


def backup_old_mapping(path, mapping):
    timestamp = datetime.timestamp(datetime.now())
    bkp_path = path + f'.{timestamp}'
    with open(bkp_path, 'w') as f:
        json.dump(mapping, f)


def update_channel_mapping(folder, plate_name, new_serum_names):
    mapping_file = os.path.join(folder, 'channel_mapping.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    new_mapping = {k: v for k, v in mapping.items() if v is None or (not v.startswith('serum'))}
    serum_mapping = {k: v for k, v in mapping.items() if v is not None and v.startswith('serum')}

    serum_names = set(serum_mapping.values())

    if serum_names != set(new_serum_names):
        assert len(serum_names) == len(new_serum_names) == 1
        serum_mapping = {k: v for k, v in zip(serum_mapping.keys(), new_serum_names)}

    new_mapping.update(serum_mapping)
    if new_mapping != mapping:
        backup_old_mapping(mapping_file, mapping)
        with open(mapping_file, 'w') as f:
            json.dump(new_mapping, f)


def update_channel_mappings(root):
    folders = glob(os.path.join(root, '*'))
    folders.sort()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        name = os.path.split(folder)[1]
        if name in ('deprecated', '20200405_test_images', 'tiny_test'):
            continue

        assert name in NEW_SERUM_CHANNES, name
        new_channel_names = NEW_SERUM_CHANNES[name]
        update_channel_mapping(folder, name, new_channel_names)


def print_channel_mapping(folder, plate_name):
    mapping_file = os.path.join(folder, 'channel_mapping.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    names = list(mapping.values())
    serum_names = [name for name in names if name is not None and name.startswith('serum')]

    print(plate_name, ':', serum_names)


def print_channel_mappings(root):
    folders = glob(os.path.join(root, '*'))
    folders.sort()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        name = os.path.split(folder)[1]
        if name == 'deprecated' or name == 'tiny_test':
            continue
        print_channel_mapping(folder, name)


if __name__ == '__main__':
    root = '/g/kreshuk/data/covid/covid-data-vibor'
    update_channel_mappings(root)
