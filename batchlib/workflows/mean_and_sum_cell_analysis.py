import time
from .cell_analysis import run_cell_analysis, workflow_summaries, DEFAULT_PLOT_NAMES
from ..analysis.merge_tables import MergeMeanAndSumTables


def modify_identifiers(identifiers):

    def modify(idf):
        return idf.replace('serum_', '').replace('sum', '')

    return [modify(idf) for idf in identifiers]


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
    identifiers = run_cell_analysis(config)

    # 3.) merge the two tables into default
    merger = MergeMeanAndSumTables()
    # merger(config.folder, config.folder, force_recompute=config.force_recompute)
    merger(config.folder, config.folder, force_recompute=True)

    # 4.) run all the usual workflow summaries
    identifiers = modify_identifiers(identifiers)
    stat_names = [idf + stat_name
                  for idf in identifiers for stat_name in DEFAULT_PLOT_NAMES]
    name = "MeanAndSumCellAnalysisWorkflow"
    workflow_summaries(name, config, t0, stat_names)
