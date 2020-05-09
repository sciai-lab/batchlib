import argparse
import os
from datetime import date

from batchlib.slack.summarize_experiments import summarize_experiments
from process_for_manuscript import all_kinder_plates, all_manuscript_plates

ROOT_OUT = '/g/kreshuk/data/covid/data-processed'


def summarize_manuscript_experiment(token, clean_up):
    plate_names = all_manuscript_plates()
    folders = [os.path.join(ROOT_OUT, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'manuscript_plates_{today}'
    summarize_experiments(folders, experiment, slack_token=token, clean_up=clean_up)


def summarize_kinder_experiment(token, clean_up):
    plate_names = all_kinder_plates()
    folders = [os.path.join(ROOT_OUT, name) for name in plate_names]

    today = date.today().strftime('%Y%m%d')
    experiment = f'kinder_study_plates_{today}'
    summarize_experiments(folders, experiment, slack_token=token, clean_up=clean_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()
    token = args.token

    clean_up = token is not None

    summarize_kinder_experiment(token, clean_up)
    summarize_manuscript_experiment(token, clean_up)
