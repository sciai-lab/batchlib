import argparse
import os
from datetime import date
from glob import glob
import urllib.parse

from pymongo import MongoClient
from tqdm import tqdm

from batchlib.analysis.cell_level_analysis import CellLevelAnalysis
from batchlib.mongo.plate_metadata_repository import PlateMetadataRepository
from batchlib.reporting import make_and_upload_summary, SlackSummaryWriter
from batchlib.util.plate_visualizations import all_plots
from batchlib.workflows.cell_analysis import bg_dict_for_plots, default_bg_parameters
from process_for_manuscript import all_kinder_plates, all_manuscript_plates

# ROOT_OUT = '/g/kreshuk/data/covid/data-processed'
ROOT_OUT = '/g/kreshuk/data/covid/data-processed-scratch'


def summarize_manuscript_experiment(root, token, clean_up, ignore_incomplete, metadata_repository):
    plate_names = all_manuscript_plates()
    folders = [os.path.join(root, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'manuscript_plates_{today}'
    make_and_upload_summary(folders, experiment, token=token, clean_up=clean_up,
                            ignore_incomplete=ignore_incomplete,
                            metadata_repository=metadata_repository)


def summarize_kinder_experiment(root, token, clean_up, ignore_incomplete, metadata_repository):
    plate_names = all_kinder_plates()
    folders = [os.path.join(root, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'kinder_study_plates_{today}'
    make_and_upload_summary(folders, experiment, token=token, clean_up=clean_up,
                            ignore_incomplete=ignore_incomplete,
                            metadata_repository=metadata_repository)


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def redo_summary(root):
    folder_names = os.listdir(root)
    # folder_names = ['titration_plate_20200403_154849']

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
            channel_names = ['serum_IgG']
        else:
            stat_names = ['IgG_ratio_of_q0.5_of_means',
                          'IgG_robust_z_score_means',
                          'IgA_ratio_of_q0.5_of_means',
                          'IgA_robust_z_score_means']
            channel_names = ['serum_IgG', 'serum_IgA']

        table_path = CellLevelAnalysis.folder_to_table_path(folder)

        misc_folder = '../../misc'
        config = DummyConfig(input_folder=folder, misc_folder=misc_folder)
        bg_params, _ = default_bg_parameters(config, channel_names)
        bg_dict = bg_dict_for_plots(bg_params, table_path)

        all_plots(table_path, plot_folder,
                  table_key='wells/default',
                  identifier='per-well',
                  stat_names=stat_names,
                  wedge_width=0,
                  bg_dict=bg_dict)

        summary_writer = SlackSummaryWriter()
        summary_writer(folder, folder, force_recompute=True)

    for folder_name in tqdm(folder_names):
        folder = os.path.join(root, folder_name)
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
    parser.add_argument('--root', type=str, default=ROOT_OUT)

    # configure db connection
    parser.add_argument('--host', type=str, help='IP of the MongoDB primary DB', default=None)
    parser.add_argument('--port', type=int, help='MongoDB port', default=27017)
    parser.add_argument('--user', type=str, help='MongoDB user', default='covid19')
    parser.add_argument('--password', type=str, help='MongoDB password', default=None)
    parser.add_argument('--db', type=str, help='Default database', default='covid')

    args = parser.parse_args()
    token = args.token
    redo = bool(args.redo)

    # escape username and password to be URL friendly
    if args.host is None:
        metadata_repository = None
        print("Database was not specified, will not add any metadata!")
    else:
        username = urllib.parse.quote_plus(args.user)
        password = urllib.parse.quote_plus(args.password)

        mongodb_uri = f'mongodb://{username}:{password}@{args.host}:{args.port}/?authSource={args.db}'
        client = MongoClient(mongodb_uri)
        db = client[args.db]
        metadata_repository = PlateMetadataRepository(db)

    if redo:
        redo_summary(args.root)
    else:
        clean_up = token is not None
        # summarize_kinder_experiment(args.root, token, clean_up, bool(args.ignore_incomplete),
        #                             metadata_repository=metadata_repository)
        summarize_manuscript_experiment(args.root, token, clean_up, bool(args.ignore_incomplete),
                                        metadata_repository=metadata_repository)
