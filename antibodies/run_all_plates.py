#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import traceback
import os
import sys
from glob import glob

from batchlib.workflows import run_cell_analysis, cell_analysis_parser


def cell_analysis_workflow():
    parser = cell_analysis_parser('./configs', 'cell_analysis.conf')
    config = parser.parse_args()
    run_cell_analysis(config)


def run_plates(folders):
    for folder in folders:

        if not os.path.isdir(folder):
            continue
        if 'channel_mapping.json' not in os.listdir(folder):
            print(f'\nSkipping {folder} because it is missing channel_mapping.json\n')
            continue

        try:
            cell_analysis_workflow(folder, use_unique_output_folder=False)
        except Exception:
            print(f'\nException while evaluating folder {folder}.')
            print(traceback.format_exc())
            print('\ncontinuing..\n')
    print('all plates processed')


if __name__ == '__main__':
    # hack to parse all folders given as first arguments
    folders = []
    while len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        folders.append(sys.argv[1])
        del sys.argv[1]
    if len(folders) == 0:
        # default to all folders in covid-data-vibor
        in_folder = '/home/covid19/data/covid-data-vibor'
        folders = glob(in_folder + '/*')
        print(sys.argv)
    print('Processing these plates:')
    [print(folder) for folder in folders]
    print()
    print(f'Args being passed to workflows: {sys.argv}\n')

    run_plates(folders)
