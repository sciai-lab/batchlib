import argparse
import json
import os
from subprocess import check_output

DEFAULT_CONFIG = 'configs/cell_analysis_db.conf'
DEFAULT_SLACK_CONFIG = 'configs/cell_analysis_db_slack.conf'

ALT_BG_PLATES = ('plate6rep2_wp_20200507_131032_010',
                 'titration_plate_20200403_154849')


def get_bg_config_file(folder, tischi_mode):
    plate_name = os.path.split(folder)[1]
    if tischi_mode:
        if plate_name in ALT_BG_PLATES:
            return './configs/tischi_alt_bg.conf'
        else:
            return './configs/tischi_bg.conf'
    else:
        if plate_name in ALT_BG_PLATES:
            return './configs/cell_analysis_alt_bg.conf'
        else:
            return './configs/cell_analysis_bg.conf'


def check_channel_mappings(folder_list):
    for folder in folder_list:
        assert os.path.exists(os.path.join(folder, 'channel_mapping.json')), f"{folder} does not have a channel mapping"


def submit_folders(folder_list, config_file=None, fixed_background=False, tischi_mode=False, mean_and_sum=False):
    assert all(os.path.exists(folder) for folder in folder_list), str(folder_list)
    check_channel_mappings(folder_list)

    if mean_and_sum:
        assert False, "Currently not working !"
        slurm_file = 'submit_mean_and_sum.batch'
    else:
        slurm_file = 'submit_cell_analysis_var_conf.batch'
    print("Using slurm file:", slurm_file)

    for folder in folder_list:

        if fixed_background:
            this_config_file = get_bg_config_file(folder, tischi_mode)
        else:
            assert config_file is not None
            this_config_file = config_file

        print("Submitting", folder, "with config file", this_config_file)
        cmd = ['sbatch', slurm_file, this_config_file, folder]
        try:
            output = check_output(cmd).decode('utf8').rstrip('\n')
        except Exception as e:
            print("job submission failed with", str(e))
            raise e
        print(output, "for input folder", folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit list of folders to be processed to slurm")

    parser.add_argument('input_file', type=str, help="json file with list of folders to be processed")
    parser.add_argument('--config_file', type=str, default='', help='config file to be used')
    parser.add_argument('--fixed_background', type=int, default=0, help='run computation with fixed background')
    parser.add_argument('--tischi_mode', type=int, default=0, help='recompute stuff for tischi')
    parser.add_argument('--write_to_slack', type=int, default=0, help='post results to slack')
    parser.add_argument('--mean_and_sum', type=int, default=0, help='run the mean and sum workflow')

    args = parser.parse_args()
    in_file = args.input_file

    config_file = args.config_file
    fixed_background = bool(args.fixed_background)
    write_to_slack = bool(args.write_to_slack)
    tischi_mode = bool(args.tischi_mode)

    if tischi_mode:
        assert fixed_background

    if config_file != '' and fixed_background:
        raise ValueError("Invalid parameter combination")

    if config_file == '' and (not fixed_background):
        config_file = DEFAULT_SLACK_CONFIG if write_to_slack else DEFAULT_CONFIG

    with open(in_file, 'r') as f:
        folder_list = json.load(f)

    submit_folders(folder_list, config_file, fixed_background, tischi_mode, bool(args.mean_and_sum))
