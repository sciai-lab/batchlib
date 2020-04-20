#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

from glob import glob
import os
import sys
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config2
from instance_analysis_workflow2 import parser as instance_workflow_parser
from pixel_analysis_workflow1 import run_pixel_analysis1, parse_pixel_config1
from pixel_analysis_workflow1 import parser as pixel_workflow_parser


def run_instance2(in_folder, use_unique_output_folder):
    config, unknown = instance_workflow_parser().parse_known_args()
    print(f'Arguments unknown to instance workflow: {unknown}')
    config.input_folder = in_folder
    config.use_unique_output_folder = use_unique_output_folder
    run_instance_analysis2(config)


def run_pixel1(in_folder, use_unique_output_folder):
    config, unknown = pixel_workflow_parser().parse_known_args()
    print(f'Arguments unknown to pixel workflow: {unknown}')
    config.input_folder = in_folder
    config.use_unique_output_folder = use_unique_output_folder
    run_pixel_analysis1(config)


def run_plates(folders):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        if 'channel_mapping.json' not in os.listdir(folder):
            print(f'\nSkipping {folder} because it is missing channel_mapping.json\n')
            continue
        # run_all_workflows(folder)
        # TODO don't hard-code to these workflows
        try:
            run_instance2(folder, use_unique_output_folder=False)
            run_pixel1(folder, use_unique_output_folder=False)
        except Exception as e:
            print(f'\nException while evaluating folder {folder}.')
            print(e)
            raise e
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
