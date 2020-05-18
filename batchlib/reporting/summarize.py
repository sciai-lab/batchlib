import os
from glob import glob
from shutil import rmtree, copyfile, make_archive

from slack import WebClient
from .export_tables import export_scores


def make_and_upload_summary(folder_list, experiment_name, token,
                            channel='#latest-results', clean_up=True, ignore_incomplete=False,
                            metadata_repository=None):
    res_file = make_summary(folder_list, experiment_name,
                            export_score_table=True, clean_up=clean_up,
                            ignore_incomplete=ignore_incomplete,
                            metadata_repository=metadata_repository)
    if token is not None:
        upload_summary_to_slack(res_file, token, channel, clean_up=clean_up)


def upload_summary_to_slack(summary_file, token,
                            channel='#latest-results', clean_up=True):
    """ Update zipped summary folder to slack.
    """
    experiment_name = os.path.splitext(os.path.split(summary_file)[1])[0]
    client = WebClient(token=token)
    msg = f"Posting all results for {experiment_name}"
    client.files_upload(channels=channel,
                        initial_comment=msg,
                        file=summary_file, filename=experiment_name)
    if clean_up:
        os.remove(summary_file)


def make_summary(folder_list, experiment_name,
                 export_score_table=True, ignore_incomplete=False, clean_up=True,
                 metadata_repository=None):
    """ Write all experiments to zip.
    """

    tmp_folder = os.path.join(experiment_name, experiment_name)

    for folder in folder_list:
        plate_name = os.path.split(folder)[1]
        summary_folder = os.path.join(folder, 'summary')

        out_folder = os.path.join(tmp_folder, plate_name)
        os.makedirs(out_folder, exist_ok=True)
        have_summary = os.path.exists(summary_folder)

        # copy the plots
        plots = glob(os.path.join(summary_folder, '*.jpg'))
        if len(plots) == 0:
            have_summary = False
        for plot in plots:
            plot_name = os.path.split(plot)[1]
            copyfile(plot, os.path.join(out_folder, plot_name))

        # copy the tables
        tables = glob(os.path.join(summary_folder, '*.xlsx'))
        if len(tables) == 0:
            have_summary = False
        for table in tables:
            table_name = os.path.split(table)[1]
            table_out = os.path.join(out_folder, table_name)
            copyfile(table, table_out)

        # copy the text
        msg_path = os.path.join(summary_folder, 'message.md')
        if os.path.exists(msg_path):
            copyfile(msg_path, os.path.join(out_folder, "README.md"))
        else:
            have_summary = False

        if not have_summary:
            msg = f"Did not find a summary at {summary_folder} for plate {plate_name}"
            if ignore_incomplete:
                print(msg)
                rmtree(out_folder)
            else:
                raise RuntimeError(msg)

    if export_score_table:
        score_table_path = os.path.join(experiment_name, f'{experiment_name}_scores.xlsx')
        export_scores(folder_list, score_table_path, metadata_repository=metadata_repository)

    # zip the folder
    res_zip = f'{experiment_name}.zip'
    make_archive(experiment_name, 'zip', experiment_name)
    assert os.path.exists(res_zip), res_zip

    # clean up
    if clean_up:
        rmtree(tmp_folder)

    return res_zip
