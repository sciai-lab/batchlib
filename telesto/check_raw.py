import argparse
import os
from glob import glob
from random import shuffle

import imageio
import napari
import pandas as pd


def view_im(path, label=None):
    im = imageio.imread(path)
    with napari.gui_qt():
        name = '' if label is None else label
        viewer = napari.Viewer(title=name)
        viewer.add_image(im)

        for chan_id in range(3):
            chan = im[..., chan_id]
            viewer.add_image(chan, name=f'channel-{chan_id}', visible=False)


def check_train(folder, n_images, shuffle_ims=False):
    im_pattern = os.path.join(folder, 'training_set/*.png')
    table = os.path.join(folder, 'training_set/training_set.tsv')
    table = pd.read_csv(table, sep='\t')
    label_dict = dict(zip(table['id'], table['label']))

    images = glob(im_pattern)
    images.sort()
    print("Found", len(images), "images")

    if shuffle_ims:
        shuffle(images)

    ii = 0
    for im_path in images:
        im_id = os.path.split(im_path)[1]
        im_id = os.path.splitext(im_id)[0]
        view_im(im_path, label=label_dict[im_id])
        ii += 1
        if ii >= n_images:
            break


def check_test(folder, n_images, shuffle_ims=False):
    im_pattern = os.path.join(folder, 'test_set/*.png')
    images = glob(im_pattern)
    images.sort()
    print("Found", len(images), "images")

    if shuffle_ims:
        shuffle(images)

    ii = 0
    for im_path in images:
        print(im_path)
        view_im(im_path)
        ii += 1
        if ii >= n_images:
            break


if __name__ == '__main__':
    default_root = '/home/pape/Work/data/covid/telesto'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=1)
    parser.add_argument('--root', default=default_root)
    parser.add_argument('--n_images', default=5, type=int)
    parser.add_argument('--shuffle', default=0)

    args = parser.parse_args()
    root = args.root
    n_images = args.n_images
    shuffle_ims = args.shuffle

    if args.train:
        check_train(root, n_images, shuffle_ims)
    else:
        check_test(root, n_images, shuffle_ims)
