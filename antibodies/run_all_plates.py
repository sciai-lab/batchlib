#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

from glob import glob
from run_all_workflows import run_all_workflows
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config1


def run_instance2(folder, use_unique_output_folder):
    config = parse_instance_config1()
    config.folder = folder
    config.use_unique_output_folder = use_unique_output_folder
    run_instance_analysis2(config)


def run_all_plates(with_corrected=True):
    in_folder = '/home/covid19/data/covid-data-vibor'
    folders = glob(in_folder + '/*')

    for folder in folders:
        # run_all_workflows(folder)
        # TODO don't hard-code
        run_instance2(folder, use_unique_output_folder=True)


run_all_plates()
