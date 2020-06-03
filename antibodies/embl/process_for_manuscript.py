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
    name = 'MergeAnalysisTables'
    return name in status and status[name] == 'processed'


def is_processed_mean_and_sums(folder):
    status = os.path.join(folder, 'batchlib', 'MergeAnalysisTables.status')
    if not os.path.exists(status):
        return False

    with open(status) as f:
        status = json.load(f)
    return status['state'] == 'processed'


def fixed_pattern_plates(patterns):
    inputs = []
    for pattern in patterns:
        inputs.extend(glob(os.path.join(ROOT_IN, pattern)))
    folder_names = [os.path.split(folder)[1] for folder in inputs]
    folder_names.sort()
    folder_names = list(set(folder_names))
    return folder_names


def heidelberg_kinder_plates():
    patterns = ['*plateK*', '*PlateK*']
    return fixed_pattern_plates(patterns)


def tubingen_kinder_plates():
    patterns = ['*plateT*']
    return fixed_pattern_plates(patterns)


def ulm_kinder_plates():
    patterns = ['*plateU*']
    return fixed_pattern_plates(patterns)


def all_kinder_plates():
    patterns = ['*plateK*', '*PlateK*', '*plateT*', '*plateU*']
    return fixed_pattern_plates(patterns)


def all_manuscript_plates():
    '/g/kreshuk/data/covid/covid-data-vibor'
    folder_names = [  # 'titration_plate_20200403_154849',
                    '20200417_185943_790',
                    '20200417_172611_193',
                    # '20200415_150710_683',
                    # '20200406_210102_953',
                    # '2200410_145132_254',
                    # '20200406_164555_328',
                    # '20200406_222205_911',
                    '20200420_152417_316',
                    '20200420_164920_764',
                    '20200417_132123_311',
                    '20200417_152052_943']
    pattern = '*rep*'
    rep_folders = glob(os.path.join(ROOT_IN, pattern))
    rep_names = [os.path.split(folder)[1] for folder in rep_folders]

    pattern = '*IgM*'
    igm_folders = glob(os.path.join(ROOT_IN, pattern))
    rep_names.extend([os.path.split(folder)[1] for folder in igm_folders])

    exclude = ['plate1rep3_20200505_100837_821',
               'plate2rep3_20200507_094942_519',
               'plate5rep3_20200507_113530_429',
               'plate6rep2_wp_20200507_131032_010',
               'plate9_2rep1_20200506_163349_413',
               'plate1rep4_20200526_083626_191',
               'plate2rep4_20200526_101902_924',
               'plate5rep4_20200526_120907_785',
               'plate6rep4_20200526_133304_599']

    # the pattern also matches the kinder names, so we filter them out
    rep_names = list(set(rep_names) - set(all_kinder_plates()))
    rep_names = list(set(rep_names) - set(exclude))

    folder_names.extend(rep_names)
    folder_names.sort()
    return folder_names


def folders_for_manuscript(root_out, check_if_processed):
    folder_names = all_manuscript_plates()
    print("The names of all the plates to include in the manuscript:")
    print("\n".join(folder_names))

    to_process = [os.path.join(root_out, name) for name in folder_names]
    if check_if_processed:
        to_process = [os.path.join(ROOT_IN, name)
                      for name, folder in zip(folder_names, to_process)
                      if not is_processed_mean_and_sums(folder)]
    else:
        to_process = [os.path.join(ROOT_IN, name) for name in folder_names]

    print(len(to_process), "/", len(folder_names), "still need to be processed")
    return to_process


if __name__ == '__main__':
    descr = 'Get list of the folders for the manuscript that need to be processed'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("--root", type=str, default=ROOT_OUT,
                        help='the root folder with the outputs')
    parser.add_argument("--check_if_processed", type=int, default=1,
                        help='only return folders that need processing')
    args = parser.parse_args()

    to_process = folders_for_manuscript(args.root, bool(args.check_if_processed))
    with open('for_manuscript.json', 'w') as f:
        json.dump(to_process, f, indent=2)
