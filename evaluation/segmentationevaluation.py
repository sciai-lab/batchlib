import argparse
import csv
import datetime
import glob
import os
import random

import h5py
import napari

from evaluation import RESULTS
from evaluation.makestatistics import compute_stats

KEYS_TO_USE = {'marker': {'data_type': 'image', 'visible': False},
               'nuclei': {'data_type': 'image', 'visible': True},
               'serum': {'data_type': 'image', 'visible': False},
               'cell_segmentation': {'data_type': 'label', 'visible': True},
               'mask': {'data_type': 'label', 'visible': False}
               }


def write_csv(csv_path, dataset, name, value):
    with open(csv_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, name, value])


def load_stack(h5_path):
    stack = {}
    with h5py.File(h5_path, "r") as f:
        for key, value in KEYS_TO_USE.items():
            stack[key] = f[key][list(f[key].keys())[-1]][...]
    return stack


def visualize_stack(viewer, stack):
    for key, value in KEYS_TO_USE.items():
        if value['data_type'] == 'image':
            viewer.add_image(stack[key], name=key, visible=value['visible'])

        elif value['data_type'] == 'label':
            viewer.add_labels(stack[key], name=key, visible=value['visible'])

        else:
            raise NotImplementedError


def standard_key_bind(viewer, path, csv_path, value):
    dataset = path.split("/")[-2]
    name = os.path.splitext(os.path.split(path)[1])[0]

    write_csv(csv_path, dataset, name, value)

    viewer.layers.select_all()
    viewer.layers.remove_selected()


def update_viewer(path, viewer):
    _stack = load_stack(path)
    visualize_stack(viewer, _stack)


def app(h5_paths, csv_path):

    with napari.gui_qt():
        viewer = napari.Viewer()
        stack = load_stack(h5_paths[0])

        visualize_stack(viewer, stack)

        @viewer.bind_key('q')
        def exit(viewer):
            viewer.window.close()

        @viewer.bind_key('0')
        def kb_ignore(viewer):
            standard_key_bind(viewer, h5_paths[0], csv_path, '0')
            h5_paths.pop(0)

            update_viewer(h5_paths[0], viewer)

        @viewer.bind_key('1')
        def kb1(viewer):
            standard_key_bind(viewer, h5_paths[0], csv_path, '1')
            h5_paths.pop(0)

            update_viewer(h5_paths[0], viewer)

        @viewer.bind_key('2')
        def kb2(viewer):
            standard_key_bind(viewer, h5_paths[0], csv_path, '2')
            h5_paths.pop(0)

            update_viewer(h5_paths[0], viewer)

        @viewer.bind_key('3')
        def kb3(viewer):
            standard_key_bind(viewer, h5_paths[0], csv_path, '3')
            h5_paths.pop(0)

            update_viewer(h5_paths[0], viewer)

        @viewer.bind_key('4')
        def kb4(viewer):
            standard_key_bind(viewer, h5_paths[0], csv_path, '4')
            h5_paths.pop(0)

            update_viewer(h5_paths[0], viewer)


def _argparse():
    parser = argparse.ArgumentParser(description='Pmaps Quality Evaluation Script')
    parser.add_argument('--data', type=str,
                        help='Path to directory with the plates collection',
                        default='/mnt/covid19/data/data-processed-seg-new')

    parser.add_argument('--results', type=str,
                        help='defines where the results will be saved',
                        default=RESULTS)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _argparse()

    all_stacks = glob.glob(f"{args.data}/**/*.h5")
    random.shuffle(all_stacks)

    now = datetime.datetime.now()
    csv_results_path = f"{args.results}/segmentation_{now.strftime('%Y_%m_%d_%H%M%S')}.csv"

    # Write header
    with open(csv_results_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "file", "evaluation"])

    app(all_stacks, csv_results_path)

    compute_stats(args.results)


