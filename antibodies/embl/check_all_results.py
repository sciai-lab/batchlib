import os
import json
from glob import glob

ROOT_IN = '/g/kreshuk/data/covid/covid-data-vibor'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed'

EXCLUDE_NAMES = ['tiny_test', 'deprecated', 'channel_mapping.json']


def check_processed(name):
    folder = os.path.join(ROOT_OUT, name)
    status = os.path.join(folder, 'batchlib', 'CellAnalysisWorkflow.status')

    # TODO implement better way to check if the job was successfull
    with open(status) as f:
        status = json.load(f)
    return 'FindInfectedCells' in status


def check_all_results():
    all_folders = glob(os.path.join(ROOT_IN, '*'))
    all_names = [os.path.split(folder)[1] for folder in all_folders]
    all_names = set(name for name in all_names if name not in EXCLUDE_NAMES)

    processed_folders = set(glob(os.path.join(ROOT_OUT, '*')))
    processed_names = set(os.path.split(folder)[1] for folder in processed_folders)

    not_processed = list(all_names - processed_names)

    if len(not_processed) > 0:
        print("The following plates have not been processed yet:")
        print("\n".join(not_processed))
    else:
        print("All output folders are present already")

    processed_names = list(processed_names)
    incorrectly_processed = [name for name in processed_names if not check_processed(name)]

    if len(incorrectly_processed) > 0:
        print("The following plates were not processed correctly:")
        print("\n".join(incorrectly_processed))
        print("These are ", len(incorrectly_processed), "/", len(all_names), "plates")
    else:
        print("All output folders have been processed correctly")

    incorrectly_processed_folders = [os.path.join(ROOT_IN, name) for name in incorrectly_processed]
    return incorrectly_processed_folders


if __name__ == '__main__':
    to_process = check_all_results()
    with open('left_to_process.json', 'w') as f:
        json.dump(to_process, f)
