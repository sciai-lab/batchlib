import os
from scipy.ndimage.morphology import binary_opening, binary_closing

from .ilastik_prediction import IlastikPrediction
from ..util import open_file


# TODO
# - make mask postprocessing optional
# - allow to specify <, > in the thresholding operation
# - support threhold = None -> we just save fg prediction and don't make binary mask
class BoundaryAndMaskPrediction(IlastikPrediction):
    """ Predict ilastik project with a boundary and a mask channel.
    """
    def __init__(self, ilastik_bin, ilastik_project,
                 input_key, boundary_key, mask_key, input_pattern='*.h5',
                 boundary_channel=1, mask_channel=0, threshold=0.5, input_ndim=None,
                 **super_kwargs):
        super().__init__(ilastik_bin=ilastik_bin, ilastik_project=ilastik_project,
                         input_key=input_key, output_key=[boundary_key, mask_key],
                         input_pattern=input_pattern,
                         input_ndim=input_ndim, output_ndim=2,
                         **super_kwargs)
        self.boundary_key = boundary_key
        self.mask_key = mask_key
        self.boundary_channel = boundary_channel
        self.mask_channel = mask_channel
        self.threshold = threshold

    def save_prediction(self, in_path, out_path):
        tmp_path = in_path[:-3] + '-raw_Probabilities.h5'
        tmp_key = 'exported_data'

        # load the data and get background and boundary channels
        with open_file(tmp_path, 'r') as f:
            ds = f[tmp_key]
            data = ds[:]
        bg, bd = data[self.mask_channel], data[self.boundary_channel]

        # we assume to get background predictions and turn them into  a foreground mask
        # if we support < and >, we can also support foreground predictions
        fg_mask = (bg < self.threshold)

        # apply opening to get rid of small foreground predictions in background
        fg_mask = binary_opening(fg_mask, iterations=2)
        # apply closing to smooth the foreground mask
        fg_mask = binary_closing(fg_mask, iterations=2).astype('uint8')

        with open_file(out_path, 'a') as f:
            self.write_result(f, self.mask_key, fg_mask)
            self.write_result(f, self.boundary_key, bd)

        # clean up
        os.remove(tmp_path)
