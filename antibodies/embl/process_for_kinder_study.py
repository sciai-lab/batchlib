import argparse
import json
import os
from process_for_manuscript import all_kinder_plates, is_processed

ROOT_IN = '/g/kreshuk/data/covid/covid-data-vibor'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed'


def folders_for_kinder_study(root_out, check_if_processed):
    folder_names = all_kinder_plates()
    print("The names of all the plates to include in the kinder study:")
    print("\n".join(folder_names))

    to_process = [os.path.join(root_out, name) for name in folder_names]
    if check_if_processed:
        to_process = [os.path.join(ROOT_IN, name)
                      for name, folder in zip(folder_names, to_process)
                      if not is_processed(folder)]
    return to_process


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get list of the folders for the kinder study')
    parser.add_argument("--root", type=str, default=ROOT_OUT,
                        help='the root folder with the outputs')
    parser.add_argument("--check_if_processed", type=int, default=1,
                        help='only return folders that need processing')
    args = parser.parse_args()

    to_process = folders_for_kinder_study(args.root, bool(args.check_if_processed))
    with open('for_manuscript.json', 'w') as f:
        json.dump(to_process, f, indent=2)
