import os
from glob import glob
from shutil import rmtree

import h5py
import napari
from batchlib.preprocessing import Preprocess


ROOT = '../../data/test_data/naming_schemes'


def run():
    corrector = '../../misc/barrel_corrector.h5'
    folders = glob(os.path.join(ROOT, '*'))
    for folder in folders:
        scheme = os.path.split(folder)[1]
        job = Preprocess.from_folder(folder, barrel_corrector_path=corrector)
        job(scheme, input_folder=folder, n_jobs=2)


def check():
    names = ['./scheme1', './scheme2', './scheme3']
    data = {}
    for name in names:
        file_ = glob(os.path.join(name, '*.h5'))[0]
        with h5py.File(file_, 'r') as f:
            for k in f:
                data[name + "_" + k] = f['%s/s0' % k][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        for name, im in data.items():
            viewer.add_image(im, name=name)


# TODO
def clean_up():
    pass


# Inspect visually that preprocessing does the right thing
run()
check()
clean_up()
