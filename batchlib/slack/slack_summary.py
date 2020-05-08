import os
import shutil
from glob import glob

import pandas as pd
from slack import WebClient
from slack.errors import SlackApiError

from ..base import BatchJobOnContainer
from ..util import get_logger, open_file

# TODO double check with Roman
DEFAULT_PLOT_PATTERNS = ('*ratio_of_q0.5_of_means*per-well.png',
                         '*robust_z_score_means*per-well.png')

# DEFAULT_SLACK_CHANNEL = '#latest-results'
DEFAULT_SLACK_CHANNEL = '#random'

logger = get_logger('Workflow.BatchJob.SlackSummaryWriter')


class SlackSummaryWriter(BatchJobOnContainer):
    def __init__(self, slack_token=None, slack_channel=DEFAULT_SLACK_CHANNEL,
                 plot_patterns=DEFAULT_PLOT_PATTERNS, **super_kwargs):

        self.summary_table_key = 'wells/default'
        self.parameter_table_key = 'plate/analysis_parameter'

        self.slack_token = slack_token
        self.slack_channel = slack_channel
        self.plot_patterns = plot_patterns

        self.write_to_slack = slack_token is not None

        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=[self.summary_table_key,
                                    self.parameter_table_key],
                         input_format=['table', 'table'],
                         **super_kwargs)

    @property
    def out_folder(self):
        return os.path.join(self.folder, 'summary')

    def check_output(self, path, **kwargs):
        if self.write_to_slack:
            check_path = os.path.join(self.out_folder, 'message_sent.txt')
        else:
            check_path = os.path.join(self.out_folder, 'message.txt')
        return os.path.exists(check_path)

    def validate_output(self, path, **kwargs):
        return self.check_output(path)

    def copy_plots(self, plot_folder, out_folder):
        for pattern in self.plot_patterns:
            plots = glob(os.path.join(plot_folder, pattern))
            for plot in plots:
                plot_name = os.path.split(plot)[1].replace('q0.5', 'median')
                plot_out = os.path.join(out_folder, plot_name)
                shutil.copyfile(plot, plot_out)

    def copy_table(self, input_file, out_path):
        with open_file(input_file, 'r') as f:
            columns, table = self.read_table(f, self.summary_table_key)
        table = pd.DataFrame(table, columns=columns)
        table.to_csv(out_path, index=False)

    # TODO need to put more info in the message:
    # - what's the background value ?
    # - is it computed or was it pre-set ?
    def compile_message(self, out_folder, plate_name, runtime):
        msg_path = os.path.join(self.out_folder, 'message.txt')
        with open(msg_path, 'w') as f:
            f.write("Results for plate %s:\n" % plate_name)
            # TODO write where the results are
            # f.write()
            if runtime is not None:
                # TODO format to hours
                f.write("Ran in %f s \n" % runtime)

    # TODO
    # - need to format the message nicer using block kit, see
    #   https://slack.dev/python-slackclient/basic_usage.html
    # - need to post the actual message
    # - check if we can get to format .csv properly or try posting as excel
    def post_slack(self, out_folder):
        msg_sent_path = os.path.join(self.out_folder, 'message_sent.txt')
        upload_patterns = ('*.png', '*.csv')

        # get slack client and connect to the workspace
        client = WebClient(token=self.slack_token)

        # post summary to slack
        try:
            logger.info(f"{self.name}: successfully connected to workspace")
            response = client.chat_postMessage(channel=self.slack_channel,
                                               text="Hello world!")
            logger.debug(f"{self.name}: posted {response} to {self.slack_channel}")

            upload_files = []
            for pattern in upload_patterns:
                to_upload = glob(os.path.join(out_folder, pattern))
                to_upload.sort()
                upload_files.extend(to_upload)

            logger.info(f"{self.name}: posting the files {upload_files}")

            for ff in upload_files:
                response = client.files_upload(channels=self.slack_channel, file=ff)
                logger.debug(f"{self.name}: posted {response} to {self.slack_channel}")

            # we just write this file if the slack message was sent
            with open(msg_sent_path, 'w'):
                pass

        except SlackApiError as e:
            logger.info(f"{self.name}: connection failed with {e.response['error']}")

    def run(self, input_files, output_files, runtime=None):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        plate_name = os.path.split(self.folder)[1]
        # make the output summary folder
        out_folder = self.out_folder
        os.makedirs(out_folder, exist_ok=True)
        logger.info(f"{self.name}: created summary folder {out_folder}")

        # copy the relevant plots to the summary folder
        plot_folder = os.path.join(self.folder, 'plots')
        self.copy_plots(plot_folder, out_folder)

        # copy the well table as csv to the summary folder
        table_out_path = os.path.join(out_folder, '%s_summary.csv' % plate_name)
        self.copy_table(input_files[0], table_out_path)

        # compile the slack message
        self.compile_message(out_folder, plate_name, runtime)

        # post to slack with the slack bot
        if self.slack_token is None:
            logger.info(f"{self.name}: no slack token given")
        else:
            self.post_slack(out_folder)
