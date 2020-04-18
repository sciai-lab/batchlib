import json
import os
from concurrent import futures

import numpy as np
from tqdm import tqdm

from batchlib.util.logging import get_logger
from ..base import BatchJobWithSubfolder
from ..util.io import open_file
from ..util.plate_visualizations import well_plot

logger = get_logger('Workflow.BatchJob.PixelAnalysis')


def compute_weighted_serum(mask, intensity, q):
    if np.sum(mask) == 0:
        return 0.

    if q == "mean":
        return float(np.mean(intensity[mask]))
    else:
        return float(np.quantile(intensity[mask], q))


def ratio(a, b):
    if b != 0:
        return a / b
    else:
        return 0.


def difference_over_sum(a, b):
    if (a + b) != 0:
        return (a - b) / (a + b)
    else:
        return 0.

def get_colorbar_range(key):
    colorbar_range = None

    if key == "ratio_of_mean_over_mean":
        colorbar_range = (1, 1.3)

    if key == "plates_ratio_of_mean_over_mean_median":
        colorbar_range = (1, 1.3)

    return colorbar_range

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
                  print_medians=True,
                  colorbar_range=get_colorbar_range(key),
                  outfile=outfile,
                  title=out_path + "\n" + key)

        outfile = os.path.join(root_path, f"plates_{key}_median.png")

        well_plot(ratios_per_file,
                  figsize=(14, 6),
                  outfile=outfile,
                  print_medians=True,
                  colorbar_range=get_colorbar_range(key),
                  title=out_path + "\n" + key)


class PixellevelAnalysis(BatchJobWithSubfolder):
    """
    """

    def __init__(self,
                 serum_key='serum',
                 infected_key='local_infected',
                 not_infected_key='local_not_infected',
                 input_pattern='*.h5',
                 output_folder="pixelwise_analysis",
                 identifier=None):
        self.serum_key = serum_key
        self.infected_key = infected_key
        self.not_infected_key = not_infected_key

        # all inputs should be 2d
        input_ndim = (2, 2, 2)

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.json'

        super().__init__(input_pattern,
                         output_ext=output_ext,
                         output_folder=output_folder,
                         input_key=[self.serum_key,
                                    self.infected_key,
                                    self.not_infected_key],
                         input_ndim=input_ndim)
        self.identifier = identifier

    def load_sample(self, path):
        with open_file(path, mode='r') as f:
            serum = self.read_input(f, self.serum_key)
            infected = self.read_input(f, self.infected_key)
            not_infected = self.read_input(f, self.not_infected_key)

        infected = infected > 0.5
        not_infected = not_infected > 0.5

        return infected, not_infected, serum

    def all_stats(self, input_file, output_file):

        infected, not_infected, serum = self.load_sample(input_file)
        result = {}

        infected_serum_intensity = compute_weighted_serum(infected, serum, "mean")
        not_infected_serum_intensity = compute_weighted_serum(not_infected, serum, "mean")

        result["ratio_of_mean_over_mean"] = ratio(infected_serum_intensity,
                                                  not_infected_serum_intensity)
        result["dos_of_mean_over_mean"] = difference_over_sum(infected_serum_intensity,
                                                              not_infected_serum_intensity)

        # compute statistics for different choices of quantiles (q = 0.5 == median)
        for q in [0.5]:
            infected_serum_intensity = compute_weighted_serum(infected, serum, q)
            not_infected_serum_intensity = compute_weighted_serum(not_infected, serum, 1 - q)
            result[f"ratio_of_q{q:0.1f}_over_q{(1-q):0.1f}"] = ratio(infected_serum_intensity,
                                                                     not_infected_serum_intensity)
            result[f"dos_of_q{q:0.1f}_over_q{(1-q):0.1f}"] = difference_over_sum(infected_serum_intensity,
                                                                                 not_infected_serum_intensity)

        with open(output_file, 'w') as fp:
            json.dump(result, fp)

    def run(self, input_files, output_files, n_jobs=1):
        logger.info("Compute stats for %i input images" % (len(input_files),))

        # compute all pixel-level stats
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(self.all_stats, input_files, output_files),
                      total=len(input_files)))
