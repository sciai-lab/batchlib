# BatchLib

Batch processing for image-analysis on high-throughput screening data; developed for covid19 antibody test set-up.


## Installation

- install the conda environment via `conda env create -f environment-gpu.yaml` or `conda env create -f environment-gpu.yaml -n custom_env_name`.
- activate the environment and install `batchlib` using `setup.py`, e.g. via running `pip install -e .` in this directory to install in development mode
- to check your installation, go to the `antibodies` directory and run the following example:
``` sh
python instance_analysis_workflow2.py -c configs/test_instance_analysis_2.conf --root /path/to/antibodies-nuclei
```

This should run through without throwing an error and create a folder `test` locally containing 9 h5 files with the results per image.
Note: until we solve issue #14, you will need to copy some test data to `data/test_inputs` and get the `antibodies-nuclei` repository.


## Usage & Development

### Run Analysis Workflows

The analysis workflows for antibody screening are in the folder `antibodies`.
For now, we have three workflows:
- `pixel_analysis_worfklow1`: pixel prediction based analysis workflow
- `instance_analysis_worfklow1`: instance segmentation based analysis workflow, using ilastik predictions
- `instance_analysis_worfklow2`: instance segmentation based analysis workflow, using network predictions

These scripts use `configargparse` to read options from a config file and enable over-riding options
from the command line. The default configurations to run on ialgpu03 are in `antibodies/configs`.

### Design Principles

`batchlib` operates on a single folder containing the data to process for a given experiment (usually all images from one plate).
Its workflows consist of indvidual jobs that apply an operation to files matching a pattern.
Jobs can be rerun on a folder once more data has been added and will only process new files.

TODO explain advanced stuff:
- multiple workflow sharing job on same folder (locking, but needs n5)
- different execution modes, `force_recompute`, `ignore_invalid_inputs`, `ignore_failed_outputs`

### Implement a new Batch Job

- Inherit from `batchlib.BatchJob` or batchlib.BatchJobOnContainer`
- Implement `self.run` with function signature `run(self, input_files, output_files, **kwargs)`
- Constraints:
    - Output should be a single file, if you need multiple files make a sub-directory and store them in there.
    - For image data, intermediate formats are either `h5` or `n5`.
    - Use `batchlib.io.open_file` in your job instead of `h5py.File` to support both `h5` and `n5`
    - Jobs should always be runnable with cpu only and should default to running on the cpu. gpu support should be activated via kwarg in run method.
