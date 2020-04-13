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


def well_plot(data_dict, fig=None, ax=None, title=None, outfile=None,
              sort=False, print_medians=True, figsize=(7.1, 4), radius=0.45, wedge_width=0.2):
    """
    data_dict should be a dictionary mapping filenames to test results (or whatever other value to visualize)
    """
    # group wells
    per_well_dict = defaultdict(list)
    for filename, value in data_dict.items():
        per_well_dict[get_well(filename)].append(value)
    # convert to dict of numpy arrays
    per_well_dict = {key: np.array(values) for key, values in per_well_dict.items()}

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
            patches.append(Wedge(center, radius, 360 / n_samples * i, 360 / n_samples * (i + 1), width=wedge_width))
            patch_values.append(value)

    if fig is None or ax is None:
        assert fig is None and ax is None, f'Please specify either neither or both fig and ax'
        fig, ax = plt.subplots(figsize=figsize)

    coll = PatchCollection(patches)
    coll.set_array(np.array(patch_values))

    ax.add_collection(coll)
    fig.colorbar(coll, ax=ax)

    if print_medians:
        for well_position, values in per_well_dict.items():
            center = well_position[1], 7 - well_position[0]
            median = np.median(values)
            plt.annotate(f'{median:4.2f}', center, ha='center', va='center')

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
              sort=True, wedge_width=0, title='test_title')
    plt.show()
