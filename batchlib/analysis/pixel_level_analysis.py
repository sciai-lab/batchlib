import json
from concurrent import futures

import numpy as np
from tqdm import tqdm
import scipy.stats

from batchlib.util.logger import get_logger
from ..base import BatchJobWithSubfolder
from ..util.io import open_file

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


class PixellevelAnalysis(BatchJobWithSubfolder):
    """
    """

    def __init__(self,
                 serum_key='serum',
                 infected_key='local_infected',
                 not_infected_key='local_not_infected',
                 output_folder="pixelwise_analysis",
                 identifier=None):

        self.serum_key = serum_key
        self.infected_key = infected_key
        self.not_infected_key = not_infected_key

        # all inputs should be 2d
        input_ndim = (2, 2, 2)

        # identifier allows to run different instances of this job on the same folder
        output_ext = '.json'

        super().__init__(output_ext=output_ext,
                         output_folder=output_folder,
                         input_key=[self.serum_key,
                                    self.infected_key,
                                    self.not_infected_key],
                         input_ndim=input_ndim)

        self.identifier = identifier

    def load_sample(self, path):
        with open_file(path, mode='r') as f:
            serum = self.read_image(f, self.serum_key)
            infected = self.read_image(f, self.infected_key)
            not_infected = self.read_image(f, self.not_infected_key)

            # TODO: this is generated as part of the segmentation analysis
            # needs to become part of the pixel computation pipeline
            if "mask" in f:
                background_mask = self.read_image(f, "mask") == 0
                background_intensity = serum[background_mask].mean()
            else:
                background_intensity = 0

        infected = infected > 0.5
        not_infected = not_infected > 0.5

        return infected, not_infected, serum, background_intensity

    def all_stats(self, input_file, output_file):

        infected, not_infected, serum, bg_intensity = self.load_sample(input_file)
        result = {}

        infected_serum_intensity = compute_weighted_serum(infected, serum, "mean")
        not_infected_serum_intensity = compute_weighted_serum(not_infected, serum, "mean")

        # save intensities
        result["infected_serum_intensity"] = float(infected_serum_intensity)
        result["not_infected_serum_intensity"] = float(not_infected_serum_intensity)
        result["background"] = float(bg_intensity)

        # compute some propper statistic tests
        if infected.sum() > 10 and not_infected.sum() > 10:
            ks_v, ks_p = scipy.stats.ks_2samp(serum[infected], serum[not_infected])
            result["ks_test_KS_value"] = ks_v
            result["ks_test_p_value"] = ks_p

            ttest_v, ttest_p = scipy.stats.ttest_ind(serum[infected], serum[not_infected], equal_var=False)
            result["ttest_test_ttest_value"] = ttest_v
            result["ttest_test_p_value"] = ttest_p

            ranksum_v, ranksum_p = scipy.stats.ranksums(serum[infected], serum[not_infected])
            result["ranksum_test_ranksum_value"] = ranksum_v
            result["ranksum_test_p_value"] = ranksum_p

        for correct_background in [True, False]:

            inf_intensity = result["infected_serum_intensity"]
            not_inf_intensity = result["not_infected_serum_intensity"]
            suffix = ""

            if correct_background:
                inf_intensity = inf_intensity - result["background"]
                not_inf_intensity = not_inf_intensity - result["background"]
                suffix += "_bgsub"

            # compute statistics
            result[f"ratio_of_mean_over_mean{suffix}"] = ratio(inf_intensity,
                                                               not_inf_intensity)
            result[f"dos_of_mean_over_mean{suffix}"] = difference_over_sum(inf_intensity,
                                                                           not_inf_intensity)

            # # compute statistics for different choices of quantiles (q = 0.5 == median)
            # for q in [0.5]:
            #     infected_serum_intensity = compute_weighted_serum(infected, serum, q)
            #     not_infected_serum_intensity = compute_weighted_serum(not_infected, serum, 1 - q)
            #     result[f"ratio_of_q{q:0.1f}_over_q{(1-q):0.1f}_{suffix}"] = ratio(infected_serum_intensity,
            #                                                                       not_infected_serum_intensity)
            #     result[f"dos_of_q{q:0.1f}_over_q{(1-q):0.1f}_{suffix}"] = difference_over_sum(infected_serum_intensity,
            #                                                                                   not_infected_serum_intensity)

        with open(output_file, 'w') as fp:
            json.dump(result, fp)

    def run(self, input_files, output_files, n_jobs=1):
        logger.info("Compute stats for %i input images" % (len(input_files),))

        # compute all pixel-level stats
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(self.all_stats, input_files, output_files),
                      total=len(input_files)))
