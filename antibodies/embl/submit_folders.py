import argparse
import json
import os
from subprocess import check_output


def submit_folders(folder_list):
    assert all(os.path.exists(folder) for folder in folder_list), str(folder_list)
    for folder in folder_list:
        cmd = ['sbatch', 'submit_cell_analysis.batch', folder]
        try:
            output = check_output(cmd).decode('utf8').rstrip('\n')
        except Exception as e:
            print("job submission failed with", str(e))
            raise e
        print(output, "for input folder", folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit list of folders to be processed to slurm")
    parser.add_argument('input_file', type=str, help="json file with list of folders to be processed")
    args = parser.parse_args()
    in_file = args.in_file

    with open(in_file, 'r') as f:
        folder_list = json.load(f)

    submit_folders(folder_list)
