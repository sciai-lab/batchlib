import os
import json
import sys
from subprocess import check_output


def submit_folders(folder_list):
    assert all(os.path.exists(folder) for folder in folder_list)
    for folder in folder_list:
        output = check_output(folder)
        print(output)


# TODO better argument parsing
if __name__ == '__main__':
    in_file = sys.argv[1]
    with open(in_file) as f:
        folder_list = json.load(f)
    submit_folders(folder_list)
