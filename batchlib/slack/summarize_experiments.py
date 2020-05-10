import os
from glob import glob
from shutil import rmtree, copyfile, make_archive

import pandas as pd
from slack import WebClient


def summarize_experiments(folder_list, experiment_name,
                          slack_token=None, slack_channel='#latest-results',
                          ignore_incomplete=False, clean_up=True):
    """ Write all experiments to zip.
    """

    tmp_folder = experiment_name

    for folder in folder_list:
        plate_name = os.path.split(folder)[1]
        summary_folder = os.path.join(folder, 'summary')

        out_folder = os.path.join(tmp_folder, plate_name)
        os.makedirs(out_folder, exist_ok=True)
        have_summary = os.path.exists(summary_folder)

        # copy the plots
        # TODO switch to jpg
        plots = glob(os.path.join(summary_folder, '*.png'))
        if len(plots) == 0:
            have_summary = False
        for plot in plots:
            plot_name = os.path.split(plot)[1]
            copyfile(plot, os.path.join(out_folder, plot_name))

        # copy the tables
        # TODO switch to excel from the start
        tables = glob(os.path.join(summary_folder, '*.csv'))
        if len(tables) == 0:
            have_summary = False
        for table in tables:
            table_name = os.path.splitext(os.path.split(table)[1])[0]
            table_out = os.path.join(out_folder, table_name + '.xlsx')

            df = pd.read_csv(table)
            df.to_excel(table_out, index=False)

            # copyfile(table, table_out)

        # copy the text
        msg_path = os.path.join(summary_folder, 'message.txt')
        if os.path.exists(msg_path):
            copyfile(msg_path, os.path.join(out_folder, "README.txt"))
        else:
            have_summary = False

        if not have_summary:
            msg = f"Did not find a summary at {summary_folder} for plate {plate_name}"
            if ignore_incomplete:
                print(msg)
                rmtree(out_folder)
            else:
                raise RuntimeError(msg)

    # zip the folder
    res_zip = f'{experiment_name}.zip'
    make_archive(experiment_name, 'zip', tmp_folder)
    assert os.path.exists(res_zip)

    # upload it
    if slack_token is not None:
        client = WebClient(token=slack_token)
        msg = f"Posting all results for {experiment_name}"
        client.files_upload(channels=slack_channel,
                            initial_comment=msg,
                            file=res_zip, filename=res_zip)

    # clean up
    if clean_up:
        os.remove(res_zip)
        rmtree(tmp_folder)
