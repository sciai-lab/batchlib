# BatchLib

Batch processing for image-analysis on high-throughput screening data; developed for covid19 antibody test set-up.

## Run Analysis Workflows

The analysis for antibody screening are in `antibodies`. To run them on the IALGPU03, give the path to the input folder and pass optional paramers, e.g.:
```sh
./pixel_analysis_workflow1.py /path/to/input-folder
```

## Installation

## Usage & Development

`batchlib` operates on a single folder containing the data to process in this batch.
Its workflows consist of indvidual jobs that apply an operation to files matching a pattern.
In the default execution mode, jobs can be rerun on a folder once more data has been added and will only
apply the operation to files that were not processed yet.

TODO advanced stuff: multiple workflow sharing job on same folder (locking, but needs n5)

### Implementing a batch job

- Inherit from `batchlib.BatchJob` or batchlib.BatchJobOnContainer`
- Initialize member `self.runners = {"default": self.run}`
- Implement `self.run` with function signature ...
- Constraints:
    - Output should be a single file, if you need multiple files make a sub-directory and store them in there
    - For image data, intermediate formats are either `h5` or `n5`. If you want to run multiple 
    - Use `batchlib.io.open_file` in your job instead of `h5py.File` to support both `h5` and `n5`
