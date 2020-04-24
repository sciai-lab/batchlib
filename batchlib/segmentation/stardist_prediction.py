import os
import numpy as np
from concurrent import futures
from functools import partial
from tqdm import tqdm

from batchlib.base import BatchJobOnContainer
from batchlib.util import open_file, normalize_percentile


def limit_gpu_memory(fraction, allow_growth=False):
    import tensorflow as tf
    from keras import backend as K
    config = tf.ConfigProto()
    if fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.allow_growth = bool(allow_growth)
    session = tf.Session(config=config)
    K.tensorflow_backend.set_session(session)


class StardistPrediction(BatchJobOnContainer):
    """
    """

    def __init__(self, model_root, model_name,
                 input_key, output_key,
                 input_channel=None, **super_kwargs):
        self.input_channel = input_channel
        input_ndim = 2 if self.input_channel is None else 3
        super().__init__(input_key=input_key, output_key=output_key,
                         input_ndim=input_ndim, output_ndim=2,
                         **super_kwargs)

        self.model_root = model_root
        self.model_name = model_name

    # TODO can we stack images along the batch axis?
    def predict_image(self, in_path, out_path, model):
        with open_file(in_path, 'r') as f:
            im = self.read_input(f, self.input_key, channel=self.input_channel)

        im = normalize_percentile(im, 1, 99.8)
        prob, dist = model.predict(im)

        # save to temporary file
        tmp_path_prob = os.path.splitext(out_path)[0] + '_prob.npy'
        tmp_path_dist = os.path.splitext(out_path)[0] + '_dist.npy'
        np.save(tmp_path_prob, prob)
        np.save(tmp_path_dist, dist)

    def segment_image(self, out_path, model):
        tmp_path_prob = os.path.splitext(out_path)[0] + '_prob.npy'
        tmp_path_dist = os.path.splitext(out_path)[0] + '_dist.npy'

        prob = np.load(tmp_path_prob)
        dist = np.load(tmp_path_dist)
        shape = prob.shape

        labels = model._instances_from_prediction(shape, prob, dist)[0]
        labels = labels.astype('uint32')
        with open_file(out_path, 'a') as f:
            self.write_result(f, self.output_key, labels)

        os.remove(tmp_path_prob)
        os.remove(tmp_path_dist)

    def run(self, input_files, output_files, gpu_id=None, n_jobs=1):

        # set number of OMP threads to 1, so we can properly parallelize over
        # the segmentation via NMS properly
        os.environ["OMP_NUM_THREADS"] = "1"

        # set additional env variables for gpu / cpu
        if gpu_id is None:
            # need to do this for the conda tensorflow cpu version
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # limit the gpu memory demand, so we can run tasks with pytorch later
            # (otherwise tf will block all gpu memory for the rest of the python process)
            limit_gpu_memory(.25)

        from stardist.models import StarDist2D
        model = StarDist2D(None, name=self.model_name, basedir=self.model_root)

        # run prediction for all images
        for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
            self.predict_image(in_path, out_path, model)

        # run segmentation for all images
        _segment = partial(self.segment_image, model=model)
        with futures.ThreadPoolExecutor(n_jobs) as tp:
            list(tqdm(tp.map(_segment, output_files), total=len(output_files)))
