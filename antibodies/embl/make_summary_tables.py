import argparse
import os
from concurrent import futures
from functools import partial
from glob import glob

from tqdm import tqdm

from batchlib.analysis.merge_tables import MergeAnalysisTables
from batchlib.reporting.export_tables import export_tables_for_plate
from batchlib.preprocessing import get_serum_keys


def make_summary_tables(folder, redo_tables, marker_name, cell_seg_name):

    table_identifiers = get_serum_keys(folder.replace('data-processed', 'covid-data-vibor'))
    if len(table_identifiers) == 1:
        reference_table_name = table_identifiers[0]
    else:
        reference_table_name = [table_id for table_id in table_identifiers if 'IgG' in table_id]
        assert len(reference_table_name) == 1, f"{table_identifiers}"
        reference_table_name = reference_table_name[0]

    if redo_tables:
        job = MergeAnalysisTables(table_identifiers, reference_table_name)
        job(folder, force_recompute=True)

    export_tables_for_plate(folder,
                            cell_table_name=cell_seg_name,
                            marker_name=marker_name,
                            skip_existing=False)


def make_all_tables(redo_tables, n_jobs, marker_name, cell_seg_name):
    # TODO allow to change this from the command line
    folders = glob(os.path.join('/g/kreshuk/data/covid/data-processed/*'))

    _export = partial(make_summary_tables, redo_tables=redo_tables,
                      marker_name=marker_name, cell_seg_name=cell_seg_name)

    with futures.ProcessPoolExecutor(n_jobs) as tp:
        list(tqdm(tp.map(_export, folders), total=len(folders)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--redo_tables', default=0, type=int)
    parser.add_argument('--n_jobs', default=24, type=int)
    parser.add_argument('--marker_name', type=str, default='marker_for_infected_classification')
    parser.add_argument('--cell_seg_name', type=str, default='cell_segmentation')

    args = parser.parse_args()
    make_all_tables(bool(args.redo_tables), args.n_jobs,
                    args.marker_name, args.cell_seg_name)
