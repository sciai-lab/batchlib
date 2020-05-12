import argparse
import os
from datetime import date
from glob import glob

from tqdm import tqdm

from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
from batchlib.reporting import make_and_upload_summary, SlackSummaryWriter
from batchlib.util.plate_visualizations import all_plots
from process_for_manuscript import all_kinder_plates, all_manuscript_plates

ROOT_OUT = '/g/kreshuk/data/covid/data-processed'


def summarize_manuscript_experiment(token, clean_up, ignore_incomplete):
    plate_names = all_manuscript_plates()
    folders = [os.path.join(ROOT_OUT, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'manuscript_plates_{today}'
    make_and_upload_summary(folders, experiment, token=token, clean_up=clean_up,
                            ignore_incomplete=ignore_incomplete)


def summarize_kinder_experiment(token, clean_up, ignore_incomplete):
    plate_names = all_kinder_plates()
    folders = [os.path.join(ROOT_OUT, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'kinder_study_plates_{today}'
    make_and_upload_summary(folders, experiment, token=token, clean_up=clean_up,
                            ignore_incomplete=ignore_incomplete)


def redo_summary():
    folder_names = all_manuscript_plates()  # + all_kinder_plates()

    def redo_folder(folder):
        plot_folder = os.path.join(folder, 'plots')
        this_plots = glob(os.path.join(plot_folder, '*.png'))
        for pp in this_plots:
            os.remove(pp)
        this_plots = glob(os.path.join(folder, 'summary', '*.png'))
        for pp in this_plots:
            os.remove(pp)

        if 'titration' in folder or 'plate8rep2' in folder:
            stat_names = ['IgG_ratio_of_q0.5_of_means',
                          'IgG_robust_z_score_means']
        else:
            stat_names = ['IgG_ratio_of_q0.5_of_means',
                          'IgG_robust_z_score_means',
                          'IgA_ratio_of_q0.5_of_means',
                          'IgA_robust_z_score_means']

        table_path = CellLevelAnalysis.folder_to_table_path(folder)
        all_plots(table_path, plot_folder,
                  table_key='wells/default',
                  identifier='per-well',
                  stat_names=stat_names,
                  wedge_width=0)

        summary_writer = SlackSummaryWriter()
        summary_writer(folder, folder, force_recompute=True)

    for folder_name in tqdm(folder_names):
        folder = os.path.join(ROOT_OUT, folder_name)
        try:
            redo_folder(folder)
        except Exception as e:
            print(f"Raised {e} for {folder_name}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--redo', type=int, default=0)
    parser.add_argument('--ignore_incomplete', type=int, default=0)

    args = parser.parse_args()
    token = args.token
    redo = bool(args.redo)

    if redo:
        redo_summary()
    else:
        clean_up = token is not None
        # summarize_kinder_experiment(token, clean_up, bool(args.ignore_incomplete))
        summarize_manuscript_experiment(token, clean_up, bool(args.ignore_incomplete))
