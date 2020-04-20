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
Note: until we solve everything listed in issue #14, you will need to get the (private) `antibodies-nuclei` repository separately.


## Usage & Development

### Run Analysis Workflows

The analysis workflows for antibody screening are in the folder `antibodies`.
For now, we have three workflows:
- `pixel_analysis_worfklow1`: pixel prediction based analysis workflow
- `instance_analysis_worfklow1`: instance segmentation based analysis workflow, using ilastik predictions
- `instance_analysis_worfklow2`: instance segmentation based analysis workflow, using network predictions

These scripts use `configargparse` to read options from a config file and enable over-riding options
from the command line. The default configurations to run on ialgpu03 are in `antibodies/configs`.

### How does it work?

Workflows operate on a single folder containing the data to be processed for a given experiment (usually all images from one plate).
They consist of indvidual jobs that apply an operation to all files that match a given pattern.
Jobs can be rerun on a folder when more data has been added and will only process the new files.

There are som more advanced methods of execution, they can be activated by passing the corresponding flag to `run_workflow` or the job's `__call__` method:
- `force_recompute`: Run computation for ALL files.
- `ignore_invalid_inputs`: Don't throw an error if there are invalid input files, but continue computing on the valid ones.
- `ignore_failed_outputs`: Don't throw an error if the computation fails, but continue with the next job,

### Troubleshooting

- **Can I see the progress of my job?** All files related to running workflows are stored in the `batchlib` subfolder of the folder where the data is being processed. There is a `.status` file for the workflows and individual jobs that keep track of the progress. There is also a`.log` file that contains everything that was logged.
- **I have failed inputs or outputs.** Look into the `.status` file of the job, It will contain the paths to the files for which it failed. Fix the issues and rerun the job.
- **I try to run a workflow but it does not start.** There is a `.lock` file in the `batchlab` folder that prevents multiple workflows being run on the same folder at the same time. It might not be deleted properly if a job gets killed or segaults. Just delete it and rerun.

### Data model

The intermediate image data associated with one raw image is stored in an h5 container with the same name as the image.
All images are stored with one group per image channel. The group layout for a channel called `data` looks like this:
```
/data
  /s0
  /s1
  /s2
  ...
```
The sub-datasets `sI` store the channel's image data as multi-scale image pyramid.
In addition, the group `data` contains metadata to display the image in the [plateViewer fiji plugin](https://github.com/embl-cba/fiji-plugin-plateViewer).

### Implement a new Batch Job

- Inherit from `batchlib.BatchJob` or batchlib.BatchJobOnContainer`
- Implement `self.run` with function signature `run(self, input_files, output_files, **kwargs)`
- Constraints:
    - Output should be a single file per input file. If you need multiple files per input, create a sub-directory and store them in there.
    - For image data, intermediate formats are either `h5` or `n5`. Use the methods `read_input` / `write_result` to read / write data in the batchlib data model.
    - Use `batchlib.io.open_file` in your job instead of `h5py.File` to support both `h5` and `n5`
    - Jobs should always be runnable with cpu only and should default to running on the cpu. gpu support should be activated via kwarg in run method.

### Logging
Global log level can be passed via an environment variable `LOGLEVEL` during the workflow execution, e.g.
```sh
LOGLEVEL=DEBUG python instance_analysis_workflow2.py -c configs/test_instance_analysis_2.conf --root /path/to/antibodies-nuclei
```

The workflow logger (named `Workflow`) is where all of the file/console handlers are registered, so make sure any new
logger is a child of the `Workflow` logger, e.g. `Workflow.MyJob`. All log events in the child loggers will automatically
be propagated to the parent `Workfow` logger. As an example:
```python
log1 = get_logger('Workflow') # get root logger
add_file_handler(log1, 'work_dir', 'workflow_name')

# in the job class or somewhere else
l2 = get_logger('Workflow.Job1') # this is the child logger of the 'Workfow' logger
l2.info('some message') # the message will be propagated to all the handlers registered in the parent logger
```

