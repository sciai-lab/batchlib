import numpy as np


def normalize(input_, eps=1e-6):
    input_ = input_.astype(np.float32)
    input_ -= input_.min()
    input_ /= (input_.max() + eps)
    return input_


# maybe improve barrel correction using: https://github.com/marrlab/BaSiC
def barrel_correction(image, barrel_corrector):
    if image.shape != barrel_corrector.shape:
        raise ValueError(f'Shape mismatch: {image.shape} != {barrel_corrector.shape}')
    # cast back to uint16 to keep the same datatype
    corrected = (image / barrel_corrector).astype(image.dtype)
    return corrected
