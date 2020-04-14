from glob import glob
import os
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.pyplot import imshow

# list of all negative results (by Vibor)
plate911 = ('WellA03',
            'WellA03',
            'WellB02',
            'WellB03',
            'WellB04',
            'WellB05',
            'WellC02',
            'WellC03',
            'WellE02',
            'WellE04',
            'WellE05',
            'WellF02',
            'WellF03',
            'WellF04',
            'WellG02',
            'WellG03',
            'WellG04',
            'WellH02',
            'WellH04',
            'WellH05')

plate953 = ('WellA02',
            'WellA03',
            'WellA04',
            'WellA05',
            'WellB02',
            'WellB03',
            'WellB04',
            'WellC02',
            'WellC03',
            'WellC05',
            'WellD02',
            'WellD04',
            'WellE04',
            'WellE02',
            'WellF02',
            'WellF04',
            'WellF05',
            'WellF07',
            'WellG02',
            'WellG04',
            'WellG05',
            'WellH02',
            'WellH04',
            'WellH05')


plate254 = ('WellB01',
            'WellA10',
            'WellB10',
            'WellC10',
            'WellD10',
            'WellE09',
            'WellE10',
            'WellF09',
            'WellF10',
            'WellG09',
            'WellG10')


Vibor_observation_dict = {"20200406_222205_911_PixelAnalysisWorkflow1": ("plate911", plate911),
                          "20200410_145132_254_PixelAnalysisWorkflow1": ("plate254", plate254),
                          "20200406_210102_953_PixelAnalysisWorkflow1": ("plate953", plate953)}

positive = 0
negative = 0


analysis_keys = ["ratio_of_mean_over_mean",
                 "dos_of_mean_over_mean",
                 "ratio_of_q0.5_over_q0.5",
                 "dos_of_q0.5_over_q0.5"]
analysis_dict = {}


for analysis_key in analysis_keys:
    for f in glob("/home/covid19/data/data-processed-new/*PixelAnalysisWorkflow1/pixelwise_analysis/*.json"):
        pathsplit = f.split(os.sep)
        foldername = pathsplit[-3]
        if foldername in Vibor_observation_dict:
            well = pathsplit[-1].split("_")[0]

            plate_identifier, list_of_negatives = Vibor_observation_dict[foldername]
            well_identifier = (plate_identifier, well)

            with open(f, "r") as jf:
                value = json.load(jf)[analysis_key]

            if well_identifier not in analysis_dict:
                analysis_dict[well_identifier] = {"values": [value],
                                                  "negative": well in list_of_negatives}
            else:
                analysis_dict[well_identifier]["values"].append(value)

    plt.figure(figsize=(14, 6))

    all_positive_values = []
    all_negative_values = []
    all_median_positive_values = []
    all_median_negative_values = []

    for wi in analysis_dict:
        if analysis_dict[wi]["negative"]:
            all_negative_values += analysis_dict[wi]["values"]
            all_median_negative_values += [np.median(analysis_dict[wi]["values"])]
        else:
            all_positive_values += analysis_dict[wi]["values"]
            all_median_positive_values += [np.median(analysis_dict[wi]["values"])]

    plt.violinplot([all_positive_values, all_negative_values, ], vert=False)
    # x = np.array([0.95,]*len(all_positive_values) + [1.95,]*len(all_negative_values))
    # plt.scatter(y, x, alpha=0.8, color='r', label='per patient ratios\nover all cells')

    # x = np.array([1.05,]*4 + [2.05,]*8)
    # y = [np.median(arr) for arr in per_image_ratios[:, i].reshape(12, -1)]
    # plt.scatter(y, x, alpha=0.8, color='g', label='per patient median of\nper image ratios')

    # plt.legend()
    # #plt.xlim(0.75, 2.5)
    plt.xlabel(f'{analysis_key}')
    plt.yticks([1, 2], ['negative', 'positive'])
    plt.title(f'Distribution of {analysis_key}')
    plt.savefig(f"meta_{analysis_key}.png")
    plt.close()

    plt.violinplot([all_median_positive_values, all_median_negative_values, ], vert=False)
    # x = np.array([0.95,]*len(all_positive_values) + [1.95,]*len(all_negative_values))
    # plt.scatter(y, x, alpha=0.8, color='r', label='per patient ratios\nover all cells')

    # x = np.array([1.05,]*4 + [2.05,]*8)
    # y = [np.median(arr) for arr in per_image_ratios[:, i].reshape(12, -1)]
    # plt.scatter(y, x, alpha=0.8, color='g', label='per patient median of\nper image ratios')

    # plt.legend()
    # #plt.xlim(0.75, 2.5)
    plt.xlabel(f'{analysis_key}')
    plt.yticks([1, 2], ['negative', 'positive'])
    plt.title(f'Distribution of {analysis_key}')
    plt.savefig(f"meta_{analysis_key}_median_of_wells.png")
    plt.close()
