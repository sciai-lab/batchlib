#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import argparse
import json
import os

from pixel_analysis_workflow1 import run_pixel_analysis1, parse_pixel_config1
from instance_analysis_workflow1 import run_instance_analysis1, parse_instance_config1
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config2


def dump_times(times, folder):
    exp_name = os.path.split(folder)[1]
    with open('runtimes_%s.json' % exp_name, 'w') as f:
        json.dump(times, f)


def load_config(folder, output_root_name, parser,
                gpu, n_cpus,
                key=None, use_unique_output_folder=False):
    config = parser()
    config.input_folder = folder
    config.output_root_name = output_root_name
    config.use_unique_output_folder = use_unique_output_folder

    config.gpu = gpu
    config.n_cpus = n_cpus

    # set a non-default analysis key
    if key is not None:
        config.in_key_analysis = key

    return config


def run_all_workflows(input_folder, gpu, n_cpus,
                      output_root_name='data-processed-new',
                      use_unique_output_folder=False,
                      serialize_times=False):

    times = {}
    config = load_config(input_folder, output_root_name, parse_pixel_config1, gpu, n_cpus,
                         use_unique_output_folder=use_unique_output_folder)
    name, rt = run_pixel_analysis1(config)
    if serialize_times:
        times[name] = rt
        dump_times(times, input_folder)

    # TODO enable running with different analysis key?
    config = load_config(input_folder, output_root_name, parse_instance_config1, gpu, n_cpus,
                         use_unique_output_folder=use_unique_output_folder)
    name, rt = run_instance_analysis1(config)
    if serialize_times:
        times[name] = rt
        dump_times(times, input_folder)

    # TODO enable running with different analysis key?
    config = load_config(input_folder, output_root_name, parse_instance_config2, gpu, n_cpus,
                         use_unique_output_folder=use_unique_output_folder)
    name, rt = run_instance_analysis2(config)
    if serialize_times:
        times[name] = rt
        dump_times(times, input_folder)


# TODO allow to over-ride the default config file
if __name__ == '__main__':
    doc = """Run all analysis workflows for a folder.
    """
    fhelp = """Folder to store the results. will default to
    /home/covid19/data/data-processed/<INPUT_FOLDER_NAME>, which will be
    overriden if this parameter is specified
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('input_folder', type=str, help='folder with input files as tifs')
    parser.add_argument('gpu', type=int, help='id of gpu for this job')
    parser.add_argument('n_cpus', type=int, help='number of cpus')

    args = parser.parse_args()
    run_all_workflows(args.input_folder, args.gpu, args.n_cpus)
