import argparse
import json
import os
from process_for_manuscript import all_kinder_plates, heidelberg_kinder_plates, tubingen_kinder_plates, is_processed

ROOT_IN = '/g/kreshuk/data/covid/covid-data-vibor'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed'


def folders_for_kinder_study(root_out, check_if_processed, city):
    if city is None:
        folder_names = all_kinder_plates()
    elif city == 'heidelberg':
        folder_names = heidelberg_kinder_plates()
    elif city == 'tubingen':
        folder_names = tubingen_kinder_plates()
    else:
        raise ValueError(f"Invalid city name {city}")

    print("The names of all the plates to include in the kinder study:")
    print("\n".join(folder_names))

    to_process = [os.path.join(root_out, name) for name in folder_names]
    if check_if_processed:
        to_process = [os.path.join(ROOT_IN, name)
                      for name, folder in zip(folder_names, to_process)
                      if not is_processed(folder)]
    else:
        to_process = [os.path.join(ROOT_IN, name) for name in folder_names]

    print(len(to_process), "/", len(folder_names), "still need to be processed")
    return to_process


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get list of the folders for the kinder study')
    parser.add_argument("--root", type=str, default=ROOT_OUT,
                        help='the root folder with the outputs')
    parser.add_argument("--check_if_processed", type=int, default=1,
                        help='only return folders that need processing')
    parser.add_argument("--city", type=str, default=None,
                        help='only return folders for a city (heidelberg, tubingen, stuttgart)')
    args = parser.parse_args()

    to_process = folders_for_kinder_study(args.root, bool(args.check_if_processed), args.city)
    with open('for_kinder_study.json', 'w') as f:
        json.dump(to_process, f, indent=2)
