#!/g/kreshuk/software/miniconda3/envs/antibodies-gpu/bin/python

import argparse
import json
import os
from subprocess import check_output


def check_channel_mappings(folder_list):
    for folder in folder_list:
        assert os.path.exists(os.path.join(folder, 'channel_mapping.json')), f"{folder} does not have a channel mapping"


def submit_folders(folder_list, config_file=None):
    assert all(os.path.exists(folder) for folder in folder_list), str(folder_list)
    check_channel_mappings(folder_list)

    slurm_file = 'submit_drug_screen.batch'
    print("Using slurm file:", slurm_file)

    for folder in folder_list:

        print("Submitting", folder, "with config file", config_file)
        cmd = ['sbatch', slurm_file, config_file, folder]
        try:
            output = check_output(cmd).decode('utf8').rstrip('\n')
        except Exception as e:
            print("job submission failed with", str(e))
            raise e
        print(output, "for input folder", folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit list of folders to be processed to slurm")
    parser.add_argument('input_file', type=str, help="json file with list of folders to be processed")
    parser.add_argument('config_file', type=str, help='config file to be used')

    args = parser.parse_args()
    in_file = args.input_file
    config_file = args.config_file

    with open(in_file, 'r') as f:
        folder_list = json.load(f)
    submit_folders(folder_list, config_file)
