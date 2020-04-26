import os
from glob import glob

import imageio
from batchlib.preprocessing import Preprocess


# TODO make it more general
def get_barrel_corrector(folder):
    barrel_corrector_path1 = '../misc/barrel_corrector.h5'
    barrel_corrector_path2 = '../misc/barrel_corrector_1024x1024.h5'

    in_path = glob(os.path.join(folder, '*.tiff'))[0]
    im = imageio.volread(in_path)
    shape = im.shape[1:]

    if shape == (1024, 1024):
        return barrel_corrector_path2
    else:
        return barrel_corrector_path1


def preprocess_folder(in_folder, out_folder, n_jobs=1):
    barrel_corrector_path = get_barrel_corrector(in_folder)

    scale_factors = [1, 2, 4, 8, 16]
    preprocess = Preprocess.from_folder(input_folder=in_folder,
                                        barrel_corrector_path=barrel_corrector_path,
                                        scale_factors=scale_factors)
    print("Start preprocessing", in_folder, "...")
    preprocess(out_folder, input_folder=in_folder, n_jobs=n_jobs)


def preprocess_all():
    root_in = '/g/kreshuk/data/covid/covid-data-vibor'
    root_out = '/g/kreshuk/data/covid/data-processed'

    in_folders = glob(os.path.join(root_in, '*'))
    n_jobs = 12

    for in_folder in in_folders:
        name = os.path.split(in_folder)[1]
        out_folder = os.path.join(root_out, name)
        preprocess_folder(in_folder, out_folder, n_jobs=n_jobs)


preprocess_all()
