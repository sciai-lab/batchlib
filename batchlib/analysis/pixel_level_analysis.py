import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.pyplot import imshow

import pathlib
import argparse
import os
import os.path
from concurrent import futures
from glob import glob
import json
import h5py
import numpy as np
from tqdm import tqdm
from time import sleep
from ..base import BatchJob, BatchJobWithSubfolder
from ..util.plate_visualizations import well_plot


def open_hdf5(filename, *args, **kwargs):
    while True:
        try:
            hdf5_file = h5py.File(filename, *args, **kwargs)
            break  # Success!
        except OSError:
            print(f"waiting on {filename}")
            sleep(5)  # Wait a bit
    return hdf5_file


def load_sample(path):
    with open_hdf5(path, mode='r') as f:
        tritc = f['raw'][2]
        local_infection_probs = f['local_infection'][()]

    infected = local_infection_probs[0] > 0.5
    not_infected = local_infection_probs[1] > 0.5

    return infected, not_infected, tritc


def create_folder(*split_path):
    new_folder = os.path.join(*split_path)
    pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)


def compute_weighted_tritc(mask, intensity, q):
    if np.sum(mask) == 0:
        return 0.

    if q == "mean":
        return float(np.mean(intensity[mask]))
    else:
        return float(np.quantile(intensity[mask], q))

# this is what should be run for each h5 file


def all_stats(input_file, output_file, analysis_folde_name="pixelwise_analysis"):

    print(input_file)
    root_path, filename = os.path.split(input_file)
    create_folder(root_path, analysis_folde_name)

    infected, not_infected, tritc = load_sample(input_file)
    result = {}

    nominator = compute_weighted_tritc(infected, tritc, "mean")
    denominator = compute_weighted_tritc(not_infected, tritc, "mean")
    result[f"ratio_of_mean_over_mean"] = nominator / denominator
    result[f"dos_of_mean_over_mean"] = (nominator - denominator) / (nominator + denominator)

    for q in [0.5]:
        nominator = compute_weighted_tritc(infected, tritc, q)
        denominator = compute_weighted_tritc(not_infected, tritc, 1 - q)
        result[f"ratio_of_q{q:0.1f}_over_q{(1-q):0.1f}"] = nominator / denominator
        result[f"dos_of_q{q:0.1f}_over_q{(1-q):0.1f}"] = nominator / denominator

    with open(output_file, 'w') as fp:
        json.dump(result, fp)


def all_plots(json_files, out_path):

    # load first json file to get list of key
    with open(json_files[0], "r") as key_file:
        keys = [k for k in json.load(key_file).keys()]  # if k.startswith("ratio_of")]

    for key in keys:

        ratios_per_well = {}
        ratios_per_file = {}

        for jf in json_files:
            well_name = jf.split("/")[-1].split("_")[0]
            file_name = jf.split("/")[-1]

            if well_name not in ratios_per_well:
                ratios_per_well[well_name] = []

            with open(jf, "r") as jf:
                ratio = json.load(jf)[key]

            if ratio != 0:
                ratios_per_well[well_name].append(float(ratio))
                ratios_per_file[file_name] = float(ratio)

        root_path = os.path.dirname(os.path.abspath(json_files[0]))
        outfile = os.path.join(root_path, f"plates_{key}.png")

        well_plot(ratios_per_file,
                  figsize=(14, 6),
                  outfile=outfile,
                  title=out_path + "\n" + key)

        outfile = os.path.join(root_path, f"plates_{key}_median.png")

        well_plot(ratios_per_file,
                  figsize=(14, 6),
                  outfile=outfile,
                  wedge_width=0,
                  title=out_path + "\n" + key)

class PixellevelAnalysis(BatchJobWithSubfolder):
    """
    """

    def __init__(self,
                 raw_key='raw',
                 infection_key='local_infection',
                 input_pattern='*.h5',
                 output_folder="pixelwise_analysis",
                 identifier=None,
                 n_jobs=1):

        self.raw_key = raw_key
        self.infection_key = infection_key

        # prediction and raw image should be 3d (2d + channels)
        input_ndim = (3, 3)

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.json'

        self.n_jobs = n_jobs

        super().__init__(input_pattern,
                         output_ext=output_ext,
                         output_folder=output_folder,
                         input_key=[self.raw_key,
                                    self.infection_key],
                         input_ndim=input_ndim)
        self.identifier = identifier
        self.runners = {'default': self.run}

    def run(self, input_files, output_files):

        def _stat(args):
            all_stats(*args)        

        print("Compute stats for %i input images" % (len(input_files), ))
        with futures.ThreadPoolExecutor(self.n_jobs) as tp:
            list(tqdm(tp.map(_stat,
                 zip(input_files, output_files)),
                 total=len(input_files)))


class PixellevelPlots(BatchJob):
    """
    """

    def __init__(self,
                 input_pattern='pixelwise_analysis/*.json',
                 output_ext="",
                 identifier=None):

        super().__init__(input_pattern,
                         output_ext=output_ext)

        self.identifier = identifier
        self.runners = {'default': self.run}

    def run(self, input_files, output_files):
        print("plot histograms for %i input jsons" % (len(input_files), ))
        output_folder = os.path.split(output_files[0])[0]
        all_plots(input_files, output_folder)

    def validate_output(self, path):
        # We do not need to check the plot files
        return True
