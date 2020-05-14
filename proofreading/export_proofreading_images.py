import h5py
import numpy as np
import skimage.color as skc
import imageio

# import matplotlib.pyplot as plt
# import napari

from glob import glob
from batchlib.util import read_image, read_table
from scipy.ndimage import convolve
from scipy.ndimage.morphology import binary_erosion


def normalize(im):
    im = im.astype('float32')
    im -= im.min()
    im /= im.max()
    return im


def quantile_normalize(im, low=.01, high=.99):
    tlow, thigh = np.quantile(im, low), np.quantile(im, high)
    im -= tlow
    im /= thigh
    return np.clip(im, 0., 1.)


def make_2d_edges(segmentation):
    """ Make 2d edges from 2d segmentation
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def erode_seg(seg, iters):

    seg_ids = np.unique(seg)[1:]
    new_seg = np.zeros_like(seg)
    for seg_id in seg_ids:
        seg_mask = binary_erosion(seg == seg_id, iterations=iters)
        new_seg[seg_mask] = seg_id
    return new_seg


def export_proofreadng_image(path):
    with h5py.File(path, 'r') as f:
        serum = normalize(read_image(f, 'serum_IgG'))
        marker = normalize(read_image(f, 'marker'))
        marker2 = quantile_normalize(read_image(f, 'marker'))
        nuclei = normalize(read_image(f, 'nuclei'))

        seg = read_image(f, 'cell_segmentation')
        _, infected_labels = read_table(f, 'infected_cell_labels')

    infected_labels = infected_labels[:, 1]
    seg_ids = np.unique(seg)
    pos_mask = np.isin(seg, seg_ids[infected_labels == 1])
    neg_mask = np.isin(seg, seg_ids[infected_labels == 2])

    pos_seg = seg.copy()
    pos_seg[~pos_mask] = 0
    pos_seg = erode_seg(pos_seg, 1)
    pos_edges = np.logical_and(pos_seg == 0, pos_mask != 0)

    neg_seg = seg.copy()
    neg_seg[~neg_mask] = 0
    neg_seg = erode_seg(neg_seg, 1)
    neg_edges = np.logical_and(neg_seg == 0, neg_mask != 0)

    raw = np.concatenate([marker[..., None], serum[..., None], nuclei[..., None]], axis=-1)
    raw2 = np.concatenate([marker2[..., None], serum[..., None], nuclei[..., None]], axis=-1)
    raw2 = skc.rgb2hsv(raw2)
    raw2[..., 1] *= 2
    raw2 = skc.hsv2rgb(raw2).clip(0, 1)

    # TODO also export the non-enhanced
    raw2[pos_edges, :] = [1, 0.6, 0]  # orange
    raw2[neg_edges, :] = [0, 1, 1]  # cyan
    imageio.imwrite(path.replace('h5', 'png'), raw2)

    # plt.imshow(raw2)
    # plt.savefig(path.replace('h5', 'png'), frameon=False)
    # plt.show()

    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(raw, rgb=True, name='raw')
    #     viewer.add_image(raw2, rgb=True, name='enhanced')
    #     viewer.add_labels(pos_edges, name='pos-edges')
    #     viewer.add_labels(neg_edges, name='neg-edges')


# TODO update
def plot_all():
    for path in glob('*.h5'):
        export_proofreadng_image(path)


if __name__ == '__main__':
    plot_all()
