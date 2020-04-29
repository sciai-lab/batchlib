import glob
import os
import subprocess
import argparse


def sync_results(all_paths):
    for path in all_paths:
        base, file = os.path.split(path)
        plate, workflow = base.split("/")[-2], base.split("/")[-1]

        if file in main_results:
            out_path = os.path.join(destination,
                                    plate,
                                    workflow)
        else:
            out_path = os.path.join(destination,
                                    plate,
                                    workflow,
                                    auxiliary_results_destination)

        os.makedirs(out_path, exist_ok=True)
        subprocess.call(['rsync', '-avP', path, f'{out_path}/{file}'])


def _argparse():
    parser = argparse.ArgumentParser(description='This script syncs all png in a summary directory')
    parser.add_argument('--data', type=str,
                        help='Path to directory with the processed plates',
                        default='/home/covid19/data/data-processed/')

    parser.add_argument('--results', type=str,
                        help='Defines where the results will be saved',
                        default='/home/covid19/data/analysis_summary/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _argparse()
    all_png_paths = glob.glob(f'{args.data}/**/**/*.png')
    destination = args.results

    # Change this list if relevant results changes
    main_results = ['plates_ratio_of_mean_over_mean_median.png',
                    'plates_ratio_of_median_of_means_median.png']

    # To avoid visual clutter results that are not in 'main_results'
    # will be saved in a sub directory
    auxiliary_results_destination = 'auxiliary_results'
    sync_results(all_png_paths)

