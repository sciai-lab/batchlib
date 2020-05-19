import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import math
from batchlib.util.io import open_file, read_table, get_column_dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge

row_letters = np.array(list('ABCDEFGH'))
letter_to_row = {letter: i for i, letter in enumerate(row_letters)}

to_rgb = mcolors.ColorConverter().to_rgb


def make_colormap(seq):
    """
    From https://stackoverflow.com/a/16836182.
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


CATEGORICAL_COLORS = (
    to_rgb('blue'), to_rgb('blue'),
    (0, 0.8, 0), (0, 0.8, 0),
    to_rgb('yellow'), to_rgb('yellow')
)
CONTINUOUS_COLORS = (
    to_rgb('blue'), to_rgb('lightblue'),
    (0, 0.8, 0), (0, 0.8, 0),
    to_rgb('lightyellow'), to_rgb('yellow')
)


def make_colormap_absolute(
    thresholds=(0.0, 0.33, 0.66, 1.0),
    colors=CONTINUOUS_COLORS,
):
    thresholds = np.array(thresholds)
    vmin = thresholds[0]
    vmax = thresholds[-1]
    thresholds = (thresholds[1:-1] - vmin) / (vmax - vmin)
    seq = [x for i in range(len(thresholds))
           for x in (colors[2*i], colors[2*i+1], thresholds[i])] + list(colors[-2:])
    cmap = make_colormap(seq)
    return cmap, (vmin, vmax)


def get_well(filename, to_numeric=True):
    # gets well coordinates. starting at 0
    if len(filename) == 3:  # if well is given as e.g. "A04"
        match = re.match('([A-H])(\d\d)', filename)
    else:
        match = re.search('Well([A-H])(\d\d)', filename)
    if not match:
        assert False, f'could not find well name in filename {filename}'
    x, y = match.groups()
    if not to_numeric:
        return x + y  # return string
    x = letter_to_row[x]
    y = int(y) - 1
    return x, y


def make_per_well_dict(data_dict, min_samples_per_well=None):
    # group wells
    per_well_dict = defaultdict(list)
    for filename, value in data_dict.items():
        per_well_dict[get_well(filename)].append(value)
    # convert to dict of numpy arrays
    per_well_dict = {key: np.array(values) for key, values in per_well_dict.items()
                     if min_samples_per_well is None or len(values) >= min_samples_per_well}
    return per_well_dict


def well_plot(data_dict, infected_list=None,
              fig=None, ax=None, title=None, outfile=None,
              sort=False, print_medians=False, figsize=(7.1, 4), colorbar_range=None, cmap=None,
              radius=0.45, wedge_width=0.2, infected_marker_width=0.05, angular_gap=0.0,
              min_samples_per_well=None):
    """
    Shows result in the structure of a 96 well plate.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping filenames to scores.
    infected_list : list, optional
        List of filenames or well-names (something that get_well() works on, like 'WellG04') of wells corresponding to
        infected patients. They will be marked with a red circle.
    fig : matplotlib.figure.Figure, optional
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    outfile : str, optional
        Path to save the plot at.
    sort : bool
        If True, the observations for each well are sorted.
    print_medians : bool
        If True, the median values for each well are overlaid as text.
    figsize : tuple
    colorbar_range : tuple, optional
        Min and max of the colorbar. Useful if multiple well plots will be compared by eye.
    radius : float
        Radius of the circles for each well
    wedge_width : float
        Width of the wedges showing the individual observations per well. Set to 0 to only show the median.
    infected_marker_width : float
        Width of the red ring around wells of infected patients.
    angular_gap : float
        Gap (in degrees) between neighboring wedges. Useful to clearly see number of observations per well.
    min_samples_per_well : int, optional
        Wells with less samples will not be displayed.
    """

    per_well_dict = make_per_well_dict(data_dict, min_samples_per_well)

    patches = []
    patch_values = []
    nan_patches = []
    for well_position, values in per_well_dict.items():
        n_samples = len(values)
        center = well_position[1], 7 - well_position[0]
        if sort:
            values = sorted(values)
        # central circle is showing the median
        central_circle = Circle(center, radius - wedge_width)
        median = np.median(values)
        if not math.isnan(median):
            patches.append(central_circle)
            patch_values.append(median)
        else:
            nan_patches.append(central_circle)

        # outer wedges show values for individual images
        if wedge_width == 0:
            continue
        for i, value in enumerate(values):
            wedge = Wedge(center, radius,
                          (360 / n_samples * (i + angular_gap)),
                          360 / n_samples * (i + 1 - angular_gap),
                          width=wedge_width)
            if not math.isnan(value):
                patches.append(wedge)
                patch_values.append(value)
            else:
                nan_patches.append(wedge)

    if fig is None or ax is None:
        assert fig is None and ax is None, f'Please specify either neither or both fig and ax'
        fig, ax = plt.subplots(figsize=figsize)

    coll = PatchCollection(patches)
    coll.set_array(np.array(patch_values))
    if colorbar_range is not None:
        coll.set_clim(*colorbar_range)
        if cmap is not None:
            coll.set_cmap(cmap)

    ax.add_collection(coll)
    fig.colorbar(coll, ax=ax)

    ax.add_collection(PatchCollection(nan_patches, facecolors='r'))

    if infected_list is not None:
        # add red circles around infected patients
        infected_marker_patches = []
        for well in infected_list:
            well_position = get_well(well)
            center = well_position[1], 7 - well_position[0]
            infected_marker_patches.append(Wedge(center, radius,
                                                 0, 360,
                                                 width=infected_marker_width))

        infected_marker_patches = PatchCollection(infected_marker_patches, facecolors='r')
        ax.add_collection(infected_marker_patches)

    if print_medians:
        for well_position, values in per_well_dict.items():
            center = well_position[1], 7 - well_position[0]
            median = np.median(values)
            t = plt.annotate(f'{median:4.3f}', center, ha='center', va='center')
            t.set_bbox(dict(edgecolor='white', facecolor='white', alpha=0.3))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.arange(12), np.arange(1, 13))
    plt.xlim(-0.7, 11.7)
    plt.ylim(-0.7, 7.7)
    plt.yticks(np.arange(len(row_letters)), reversed(row_letters))
    if title is not None:
        plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, dpi=300, quality=75, optimize=True)
        plt.close()


def score_distribution_plots(infected_values, not_infected_values, infected_medians=None, not_infected_medians=None,
                             figsize=(12, 9), title=None, outfile=None, xlim=None, binsize=0.025,
                             violin_bw_method=None):
    """
    Plot distributions of infected and non_infected scores.

    Parameters
    ----------
    infected_values : list
        List of image-wise scores of infected patients.
    not_infected_values : list
        List of image-wise scores of not infected patients.
    infected_medians : list, optional
        List of well-wise medians of scores of infected patients.
    not_infected_medians : list, optional
        List of well-wise medians of scores of not infected patients.
    figsize : tuple
    title : str, optional
    outfile : str, optional
        Path to save the plot at.
    xlim : tuple, optional
        Range of x-axis.
    binsize : float
        Binsize for plots of cumulative distributions
    violin_bw_method : object
        Bandwidth estimation method for violin plots. See bw_method at
        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.violinplot.html

    """
    x = [1]*len(not_infected_medians) + [2]*len(infected_medians)
    y = not_infected_medians + infected_medians
    fig, ax = plt.subplots(2 if infected_medians is None else 3, 1, sharex=True, figsize=figsize,
                           gridspec_kw=dict(height_ratios=(2, 1) if infected_medians is None else (2, 1, 1)))

    # per-image violins
    ax[0].violinplot([not_infected_values, infected_values], vert=False, widths=0.8,
                     bw_method=violin_bw_method)
    ax[0].set_yticks([1, 2])
    ax[0].set_yticklabels(['negative', 'positive'])

    if infected_medians is not None:
        assert not_infected_medians is not None
        # per-well scatter and violins
        ax[0].scatter(y, x, alpha=0.2, marker='o', color='r', label='per patient ratios\nover all cells')
        violin_parts = ax[0].violinplot([not_infected_medians, infected_medians], vert=False,
                                        bw_method=violin_bw_method)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('green')

    if xlim is not None:
        ax[0].set_xlim(*xlim)
    ax[0].set_title('distribution of scores')

    # per-image cumulative distribution
    ax[1].set_title(f'cell wise score cumulative distribution')
    if xlim is None:
        bins = np.arange(min(infected_values + not_infected_values),
                         max(infected_values + not_infected_values)+binsize,
                         binsize)
    else:
        bins = np.arange(xlim[0], xlim[1] + binsize, binsize)
    ax[1].hist([infected_values, not_infected_values], bins=bins, label=['positive', 'negative'],
               density=True, cumulative=True)
    ax[1].legend()

    if infected_medians is not None:
        ax[2].set_title(f'well wise score cumulative distribution')
        ax[2].hist([infected_medians, not_infected_medians], bins=bins,
                   label=['positive', 'negative'], density=True, cumulative=True)
        ax[2].legend()

    if title is not None:
        plt.suptitle(title)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        plt.close()


def get_colorbar_range(key):
    colorbar_range = None

    if key == "ratio_of_mean_over_mean":
        colorbar_range = (1, 1.3)

    if key == "plates_ratio_of_mean_over_mean_median":
        colorbar_range = (1, 1.3)

    return colorbar_range


colorbar_threshold_dict = {
    'IgA_robust_z_score_means':     (0.0, 1.8, 2.5, 5),
    'IgG_robust_z_score_means':     (0.0, 1.8, 2.5, 5),
    'IgA_robust_z_score_sums':      (0.0, 1.8, 2.5, 5),
    'IgG_robust_z_score_sums':      (0.0, 1.8, 2.5, 5),
    'IgA_ratio_of_q0.5_of_means':   (0.9, 1.8, 2.5, 5),
    'IgG_ratio_of_q0.5_of_means':   (0.9, 1.25, 1.3, 2.0),
    'IgA_ratio_of_q0.5_of_sums':    (0.9, 1.8, 2.5, 5),
    'IgG_ratio_of_q0.5_of_sums':    (0.9, 1.25, 1.3, 2.0)
}


def all_plots(table_path, out_folder, table_key, stat_names, identifier,
              outlier_table_key='wells/outliers', bg_dict=None, **well_plot_kwargs):
    if not isinstance(stat_names, (list, tuple)):
        raise ValueError(f"stat_names must be either list or tuple, got {type(stat_names)}")
    os.makedirs(out_folder, exist_ok=True)

    # load first file to get all the column names
    with open_file(table_path, 'r') as f:
        column_names, table = read_table(f, table_key)

    if column_names[0] not in ['image_name', 'well_name']:
        raise ValueError("all_plots can only be called on a table that contains the image or well statistics")

    # check that we have all the stat names
    available_stats = set(column_names[1:])
    unknown_stats = list(set(stat_names) - available_stats)
    if len(unknown_stats) > 0:
        unknown_stats = ", ".join(unknown_stats)
        raise ValueError(f"Did not find the names {unknown_stats} in the table columns")

    # get outliers. for now, only well-wise outliers are supported
    if outlier_table_key is not None:
        with open_file(table_path, 'r') as f:
            outlier_column_names, outlier_table = read_table(f, outlier_table_key)
            outlier_list = [key
                            for key, value in get_column_dict(outlier_column_names, outlier_table, 'is_outlier').items()
                            if value]

    plate_name = os.path.split(os.path.split(table_path)[0])[1]

    for name in tqdm(stat_names, desc='making plots'):
        try:
            cmap, colorbar_range = make_colormap_absolute(colorbar_threshold_dict[name])
        except KeyError:
            print(f'Warning: No colorbar thresholds specified for stat {name}')
            cmap, colorbar_range = None, None

        # 0th column is the image / well name
        image_or_well_names = [str(im_name) for im_name in table[:, 0]]
        stat_id = column_names.index(name)
        stats_per_file = dict(zip(image_or_well_names, table[:, stat_id].astype('float')))

        title = f'{plate_name}\n{name}_{identifier}'
        if bg_dict is not None:
            channel_name = 'serum_IgG' if 'IgG' in name else 'serum_IgA'
            bg_info = bg_dict[channel_name]
            title += f'\n{bg_info}'

        outfile = os.path.join(out_folder, f"{plate_name}_{name}_{identifier}.jpg")
        well_plot(stats_per_file,
                  infected_list=outlier_list,
                  print_medians=True,
                  outfile=outfile,
                  figsize=(11, 6),
                  title=title,
                  cmap=cmap,
                  colorbar_range=colorbar_range,
                  **well_plot_kwargs)


if __name__ == '__main__':
    result_dir = '/export/home/rremme/Datasets/antibodies/covid-data-vibor/20200406_210102_953/'
    well_plot({name: np.random.rand(1)[0]
               for name in os.listdir(result_dir)},
              sort=True, wedge_width=0.2, title='test_title', print_medians=True,
              figsize=(14, 8))
    plt.show()
