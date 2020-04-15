#! /home/covid19/software/miniconda3/envs/antibodies-gpu/bin/python

import json
import os

from batchlib.analysis import Summary
from pixel_analysis_workflow1 import run_pixel_analysis1, parse_pixel_config1
from instance_analysis_workflow1 import run_instance_analysis1, parse_instance_config1
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config2


def dump_times(times, folder):
    exp_name = os.path.split(folder)[1]
    with open('runtimes_%s.json' % exp_name, 'w') as f:
        json.dump(times, f)


# TODO implement over-riding this from the command line somehow without conflicting with
# argparse
def run_all_workflows(input_folder="",
                      run_pixelwise=True,
                      run_instance1=False,
                      run_instance2=True,
                      serialize_times=False):

    times = {}

    if run_pixelwise:
        config = parse_pixel_config1()
        if input_folder != "":
            config.input_folder = input_folder
        name, rt = run_pixel_analysis1(config)
        if serialize_times:
            times[name] = rt
            dump_times(times, input_folder)

    if run_instance1:
        if input_folder != "":
            config.input_folder = input_folder
        config = parse_instance_config1()
        name, rt = run_instance_analysis1(config)
        if serialize_times:
            times[name] = rt
            dump_times(times, input_folder)

    if run_instance2:
        config = parse_instance_config2()
        if input_folder != "":
            config.input_folder = input_folder
        name, rt = run_instance_analysis2(config)
        if serialize_times:
            times[name] = rt
            dump_times(times, input_folder)

    # run the summary task
    summary = Summary()
    summary(config.folder)


if __name__ == '__main__':
    run_all_workflows()
