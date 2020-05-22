import argparse
import json
import os
from concurrent import futures
from functools import partial

from tqdm import tqdm

from batchlib.analysis.merge_tables import MergeAnalysisTables
from batchlib.reporting.export_tables import export_tables_for_plate
from batchlib.preprocessing import get_serum_keys

ROOT = '/g/kreshuk/data/covid/data-processed'


def make_summary_tables(folder, root, redo_tables, marker_name, cell_seg_name):

    root_name = os.path.split(root)[1]
    in_folder = folder.replace(root_name, 'covid-data-vibor')

    table_identifiers = get_serum_keys(in_folder)
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
                            skip_existing=True)


def get_folders(root, exclude_not_processed):
    folder_names = set(os.listdir(root))

    if exclude_not_processed:
        with open('./for_manuscript.json') as f:
            not_processed = json.load(f)
        with open('./for_kinder_study.json') as f:
            not_processed.extend(json.load(f))
        not_processed = [os.path.split(not_proc)[1] for not_proc in not_processed]
        print("Excluding", not_processed)
        folder_names = list(set(folder_names) - set(not_processed))

    return [os.path.join(root, name) for name in folder_names]


def make_all_tables(root, redo_tables, n_jobs, marker_name, cell_seg_name, exclude_not_processed):
    folders = get_folders(root, exclude_not_processed)

    _export = partial(make_summary_tables, root=root, redo_tables=redo_tables,
                      marker_name=marker_name, cell_seg_name=cell_seg_name)

    with futures.ProcessPoolExecutor(n_jobs) as tp:
        list(tqdm(tp.map(_export, folders), total=len(folders)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=ROOT)
    parser.add_argument('--exclude_not_processed', type=int, default=0)
    parser.add_argument('--redo_tables', default=0, type=int)
    parser.add_argument('--n_jobs', default=24, type=int)
    parser.add_argument('--marker_name', type=str, default='marker_for_infected_classification')
    parser.add_argument('--cell_seg_name', type=str, default=None)
    # parser.add_argument('--cell_seg_name', type=str, default='cell_segmentation')

    args = parser.parse_args()
    make_all_tables(args.root, bool(args.redo_tables), args.n_jobs,
                    args.marker_name, args.cell_seg_name, bool(args.exclude_not_processed))
