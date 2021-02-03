import os
import numpy as np
import torch

from pybio.spec.utils.transformers import load_and_resolve_spec
from pybio.spec.utils import get_instance


# TODO this is missing the normalization (preprocessing)
def check_model(path):
    """ Convert model weights from format 'pytorch_state_dict' to 'torchscript'.
    """
    spec = load_and_resolve_spec(path)

    with torch.no_grad():
        print("Loading inputs and outputs:")
        # load input and expected output data
        input_data = np.load(spec.test_inputs[0]).astype('float32')
        input_data = torch.from_numpy(input_data)
        expected_output_data = np.load(spec.test_outputs[0]).astype(np.float32)
        print(input_data.shape)

        # instantiate and trace the model
        print("Predicting model")
        model = get_instance(spec)
        state = torch.load(spec.weights['pytorch_state_dict'].source, map_location='cpu')
        model.load_state_dict(state)

        # check the scripted model
        output_data = model(input_data).numpy()
        assert output_data.shape == expected_output_data.shape
        assert np.allclose(expected_output_data, output_data)
        print("Check passed")


check_model(os.path.abspath('./UNetCovidIf.model.yaml'))
