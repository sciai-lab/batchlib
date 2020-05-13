import napari
import h5py
import numpy as np
from glob import glob
from batchlib.util import read_image, read_table
from scipy.ndimage import convolve
from scipy.ndimage.morphology import binary_dilation


def normalize(im):
    im = im.astype('float32')
    im -= im.min()
    im /= im.max()
    return im


def make_2d_edges(segmentation):
    """ Make 2d edges from 2d segmentation
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def export_proofreadng_image(path):
    with h5py.File(path, 'r') as f:
        serum = normalize(read_image(f, 'serum_IgG'))
        marker = normalize(read_image(f, 'marker'))
        nuclei = normalize(read_image(f, 'nuclei'))

        seg = read_image(f, 'cell_segmentation')
        _, infected_labels = read_table(f, 'infected_cell_labels')

    infected_labels = infected_labels[:, 1]
    seg_ids = np.unique(seg)
    pos_mask = np.isin(seg, seg_ids[infected_labels == 1])
    neg_mask = np.isin(seg, seg_ids[infected_labels == 2])

    pos_seg = seg.copy()
    pos_seg[~pos_mask] = 0
    pos_edges = make_2d_edges(pos_seg)
    pos_edges = binary_dilation(pos_edges, iterations=1)

    neg_seg = seg.copy()
    neg_seg[~neg_mask] = 0
    neg_edges = make_2d_edges(neg_seg)
    neg_edges = binary_dilation(neg_edges, iterations=1)

    raw = np.concatenate([marker[..., None], serum[..., None], nuclei[..., None]], axis=-1)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, rgb=True)
        viewer.add_labels(pos_edges)
        viewer.add_labels(neg_edges)


# TODO update
def plot_all():
    for path in glob('*.h5'):
        export_proofreadng_image()


if __name__ == '__main__':
    plot_all()
