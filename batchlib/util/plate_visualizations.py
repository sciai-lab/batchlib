import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon
from collections import defaultdict

row_letters = np.array(list('ABCDEFGH'))
letter_to_row = {letter: i for i, letter in enumerate(row_letters)}


def get_well(filename, to_numeric=True):
    # gets well coordinates. starting at 0
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
              sort=False, print_medians=False, figsize=(7.1, 4), colorbar_range=None,
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
    for well_position, values in per_well_dict.items():
        n_samples = len(values)
        center = well_position[1], 7 - well_position[0]
        if sort:
            values = sorted(values)
        # central circle is showing the median
        patches.append(Circle(center, radius - wedge_width))
        patch_values.append(np.median(values))

        # outer wedges show values for individual images
        if wedge_width == 0:
            continue
        for i, value in enumerate(values):
            patches.append(Wedge(center, radius,
                                 (360 / n_samples * (i + angular_gap)),
                                 360 / n_samples * (i + 1 - angular_gap),
                                 width=wedge_width))
            patch_values.append(value)

    if fig is None or ax is None:
        assert fig is None and ax is None, f'Please specify either neither or both fig and ax'
        fig, ax = plt.subplots(figsize=figsize)

    coll = PatchCollection(patches)
    coll.set_array(np.array(patch_values))
    if colorbar_range is not None:
        coll.set_clim(*colorbar_range)

    ax.add_collection(coll)
    fig.colorbar(coll, ax=ax)

    if infected_list is not None:
        # add red circles around infected patients
        infected_marker_patches = []
        for well in infected_list:
            well_position = get_well(well)
            center = well_position[1], 7 - well_position[0]
            infected_marker_patches.append(Wedge(center, radius + wedge_width,
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
        plt.savefig(outfile)
        plt.close()


if __name__ == '__main__':
    import os
    result_dir = '/export/home/rremme/Datasets/antibodies/covid-data-vibor/20200406_210102_953/'
    well_plot({name: np.random.rand(1)[0]
               for name in os.listdir(result_dir)},
              sort=True, wedge_width=0.2, title='test_title', print_medians=True,
              figsize=(14, 8))
    plt.show()
