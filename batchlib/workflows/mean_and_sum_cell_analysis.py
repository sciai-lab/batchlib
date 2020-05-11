import time
from .cell_analysis import run_cell_analysis, workflow_summaries
from ..analysis.merge_tables import MergeMeanAndSumTables


def mean_and_sum_cell_analysis(config):
    if config.feature_identifier is not None:
        raise ValueError("Don't support a feature identifier")

    t0 = time.time()
    # 1.) run the workflow for the mean calculation, where we ignore the nuclei
    config.feature_identifier = 'mean'
    config.ignore_nuclei = True
    run_cell_analysis(config)

    # 2.) run the workflow for the sum calculation, where we don't ignore the nuclei
    config.feature_identifier = 'sum'
    config.ignore_nuclei = False
    # don't need to write summary images twice
    config.write_summary_images = False
    run_cell_analysis(config)

    # 3.) merge the two tables into default
    merger = MergeMeanAndSumTables()
    merger(config.folder, config.folder)
    return

    # TODO I think we need to over-ride the state names to make
    # it work for IgA and IgG
    # 4.) run all the usual workflow summaries
    name = "MeanAndSumCellAnalysisWorkflow"
    workflow_summaries(name, config, ['default'], t0)
