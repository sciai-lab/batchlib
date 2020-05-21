import argparse

import h5py
import napari
import numpy as np
from skimage.measure import label

from batchlib.util import read_table, write_table, has_table
from proofreading_utils import get_image_edges_and_labels


def print_the_help():
    pass


def load_labels(path):
    key = 'proofread_infected_labels'
    with h5py.File(path, 'r') as f:
        if has_table(f, key):
            print("Loading proofread labels for", path)
            _, infected_labels = read_table(f, key)
            infected_labels = infected_labels[:, 1]
        else:
            print("Did not find proofread labels for", path)
            infected_labels = None
    return infected_labels


def save_labels(path, infected_labels):
    cols = ['label_id', 'infected_label']
    n_cells = len(infected_labels)
    table = np.concatenate([np.arange(n_cells)[:, None],
                            infected_labels[:, None]], axis=1)
    with h5py.File(path, 'a') as f:
        write_table(f, 'proofread_infected_labels', cols, table, force_write=True)


def to_infected_edges(edges, seg_ids, infected_labels):
    infected_edges = np.zeros_like(edges)
    infected_ids = seg_ids[infected_labels == 1]
    control_ids = seg_ids[infected_labels == 2]
    infected_edges[np.isin(edges, infected_ids)] = 1
    infected_edges[np.isin(edges, control_ids)] = 2
    return infected_edges


def update_labels(seg, labels, infected_labels):
    # TODO this is not robust against labeling errors, where a single cell is labeled
    # (wrongly!) as infected and control
    labeled_components = label(labels)
    _, index = np.unique(labeled_components, return_index=True)
    index = np.unravel_index(index[1:], labels.shape)

    ids_to_update = seg[index]
    labels_to_updated = labels[index]

    infected_labels[ids_to_update] = labels_to_updated
    return infected_labels


# this only supports proofreading the infected / control classification
# we also want to enable correcting the segmentaiton; ideally in the same tool
def proofreading_tool(path):

    infected_labels = load_labels(path)
    if infected_labels is None:
        raw, edges, infected_labels, seg = get_image_edges_and_labels(path,
                                                                      saturation_factor=1,
                                                                      return_seg=True)
    else:
        raw, edges, _, seg = get_image_edges_and_labels(path,
                                                        saturation_factor=1,
                                                        return_seg=True)
    marker = raw[..., 0]

    seg_ids = np.unique(seg)
    assert seg_ids.shape == infected_labels.shape, f"{seg_ids.shape}, {infected_labels.shape}"

    infected_edges = to_infected_edges(edges, seg_ids, infected_labels)

    print_the_help()

    labels = np.zeros(edges.shape, dtype='uint32')
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(marker, name='marker', visible=False)
        viewer.add_labels(infected_edges, name='infected_classification')

        viewer.add_labels(labels, name='infected_correction')

        viewer.infected_labels = infected_labels

        @viewer.bind_key('u')
        def update_infected_labels(viewer):
            print("Updating labels")
            labels = viewer.layers['infected_correction'].data
            updated_labels = update_labels(seg, labels, infected_labels)
            infected_edges = to_infected_edges(edges, seg_ids, updated_labels)
            viewer.layers['infected_classification'].data = infected_edges

        @viewer.bind_key('h')
        def print_help(viewer):
            print_the_help()

        @viewer.bind_key('s')
        def save_annotations(viewer):
            print("Saving new labels")
            labels = viewer.layers['infected_correction'].data
            updated_labels = update_labels(seg, labels, infected_labels)
            save_labels(path, updated_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    path = args.path
    proofreading_tool(path)
