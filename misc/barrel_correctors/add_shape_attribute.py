import h5py


def add_shape_attribute(p, shape):
    with h5py.File(p, 'a') as f:
        f.attrs['image_shape'] = shape


add_shape_attribute('./barrel_corrector_1024x1024.h5', [1024, 1024])
add_shape_attribute('./barrel_corrector_930x1024.h5', [930, 1024])
