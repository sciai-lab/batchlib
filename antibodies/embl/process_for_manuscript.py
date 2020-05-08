import argparse
import json
import os
from glob import glob

ROOT_IN = '/g/kreshuk/data/covid/covid-data-vibor'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed'


def is_processed(folder):
    status = os.path.join(folder, 'batchlib', 'CellAnalysisWorkflow.status')
    if not os.path.exists(status):
        return False

    with open(status) as f:
        status = json.load(f)
    return 'MergeAnalysisTables' in status


def folders_for_manuscript(root_out, check_if_processed):
    folder_names = ['titration_plate_20200403_154849',
                    '20200420_152417_316',
                    '20200420_164920_764',
                    '20200417_203228_156',
                    '20200417_132123_311',
                    '20200417_152052_943']
    pattern = '*rep*'
    rep_folders = glob(os.path.join(ROOT_IN, pattern))
    rep_names = [os.path.split(folder)[1] for folder in rep_folders]

    folder_names.extend(rep_names)
    folder_names.sort()

    print("The names of all the plates to include in the manuscript:")
    print("\n".join(folder_names))

    to_process = [os.path.join(root_out, name) for name in folder_names]
    if check_if_processed:
        to_process = [os.path.join(ROOT_IN, name)
                      for name, folder in zip(folder_names, to_process) if not is_processed(folder)]
    return to_process


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get list of the folders for the manuscript that need to be processed')
    parser.add_argument("--root", type=str, default=ROOT_OUT, help='the root folder with the outputs')
    parser.add_argument("--check_if_processed", type=int, default=1, help='only return folders that need processing')
    args = parser.parse_args()

    to_process = folders_for_manuscript(args.root, bool(args.check_if_processed))
    with open('for_manuscript.json', 'w') as f:
        json.dump(to_process, f)
