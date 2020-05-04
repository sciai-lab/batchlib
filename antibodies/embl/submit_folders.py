import os
import json
import sys
from glob import glob
from subprocess import check_output

ROOT_IN = '/g/kreshuk/data/covid/covid-data-vibor'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed'

EXCLUDE_NAMES = ['tiny_test', 'deprecated', 'channel_mapping.json']


def submit_folders(folder_list):
    assert all(os.path.exists(folder) for folder in folder_list)
    for folder in folder_list:
        cmd = ['sbatch', 'submit_cell_analysis.batch', folder]
        output = check_output(cmd).decode('utf8').rstrip('\n')
        print(output, "for input folder", folder)


# TODO better argument parsing
def parse_args():
    in_file = sys.argv[1]
    with open(in_file) as f:
        folder_list = json.load(f)
    submit_folders(folder_list)


# TODO check that the processed folders actually passed all steps
def get_not_processed_folders():
    all_folders = glob(os.path.join(ROOT_IN, '*'))
    all_names = [os.path.split(folder)[1] for folder in all_folders]
    all_names = set(name for name in all_names if name not in EXCLUDE_NAMES)

    processed_folders = set(glob(os.path.join(ROOT_OUT, '*')))
    processed_names = set(os.path.split(folder)[1] for folder in processed_folders)

    not_processed = list(all_names - processed_names)

    not_processed_folders = [os.path.join(ROOT_IN, name) for name in not_processed]
    print("Folders that still need to be processed:")
    print("\n".join(not_processed_folders))

    return not_processed_folders


if __name__ == '__main__':
    not_processed_list = get_not_processed_folders()
    submit_folders(not_processed_list[:1])
