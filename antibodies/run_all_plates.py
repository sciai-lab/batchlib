#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

from glob import glob
import os
from run_all_workflows import run_all_workflows
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config2


def run_instance2(in_folder, use_unique_output_folder):
    config = parse_instance_config2()
    config.input_folder = in_folder
    config.use_unique_output_folder = use_unique_output_folder
    run_instance_analysis2(config)


def run_all_plates(with_corrected=True):
    in_folder = '/home/covid19/data/covid-data-vibor'
    folders = glob(in_folder + '/*')

    for folder in folders:
        if 'test' in folder:
            continue
        if not os.path.isdir(folder):
            continue
        if 'channel_mapping.json' not in os.listdir(folder):
            print(f'\nSkipping {folder} because it is missing channel_mapping.json\n')
            continue
        # run_all_workflows(folder)
        # TODO don't hard-code to this workflow
        try:
            run_instance2(folder, use_unique_output_folder=False)
        except Exception as e:
            print(f'\nException while evaluating folder {folder}.')
            print(e)
            print('\ncontinuing..\n')
    print('all plates processed')


run_all_plates()
