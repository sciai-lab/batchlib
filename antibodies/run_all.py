import argparse
import json
import os

from pixel_analysis_workflow1 import run_pixel_analysis1
from instance_analysis_workflow1 import run_instance_analysis1
from instance_analysis_workflow2 import run_instance_analysis2


def dump_times(times, folder):
    exp_name = os.path.split(folder)[1]
    with open('runtimes_%s.json' % exp_name, 'w') as f:
        json.dump(times, f)


def run_all(input_folder, gpu, n_cpus,
            output_root_name='data-processed-new',
            use_unique_output_folder=True):

    times = {}

    name, rt = run_pixel_analysis1(input_folder, None, n_cpus,
                                   output_root_name=output_root_name,
                                   use_unique_output_folder=use_unique_output_folder)
    times[name] = rt
    dump_times(times, input_folder)

    name, rt = run_instance_analysis1(input_folder, None, gpu, n_cpus,
                                      output_root_name=output_root_name,
                                      use_unique_output_folder=use_unique_output_folder)
    times[name] = rt
    dump_times(times, input_folder)

    name, rt = run_instance_analysis2(input_folder, None, gpu, n_cpus,
                                      output_root_name=output_root_name,
                                      use_unique_output_folder=use_unique_output_folder)
    times[name] = rt
    dump_times(times, input_folder)


if __name__ == '__main__':
    doc = """Run all analysis workflows.
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
    run_all(args.input_folder, args.gpu, args.n_cpus)
