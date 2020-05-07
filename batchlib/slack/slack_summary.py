import os
import shutil
from glob import glob

import pandas as pd
from slackclient import SlackClient

from ..base import BatchJobOnContainer
from ..util import get_logger, open_file

# TODO only copy the relevant plots, need to discuss with Roman
DEFAULT_PLOT_PATTERNS = ()

logger = get_logger('Workflow.BatchJob.SlackSummaryWriter')


# TODO also write the relevant analysis parameter to the summary
class SlackSummaryWriter(BatchJobOnContainer):
    def __init__(self, slack_token=None, plot_patterns=DEFAULT_PLOT_PATTERNS, **super_kwargs):

        self.summary_table_key = 'wells/summary'
        self.slack_token = slack_token
        self.plot_patterns = plot_patterns

        in_pattern = '*.hdf5'
        super().__init__(input_pattern=in_pattern,
                         input_key=[self.summary_table_key],
                         input_format=['table'],
                         **super_kwargs)

    @property
    def out_folder(self):
        return os.path.join(self.folder, 'summary')

    def check_output(self, path):
        msg_path = os.path.join(self.out_folder, 'message.txt')
        return os.path.exists(msg_path)

    def validate_output(self, path):
        return self.check_output(path)

    def copy_plots(self, plot_folder, out_folder):
        plots = glob(os.path.join(plot_folder, '*.png'))
        for plot in plots:
            plot_name = os.path.split(plot)[1]
            if any(pattern in plot_name for pattern in self.plot_patterns):
                plot_out = os.path.join(out_folder, plot_name)
                shutil.copyfile(plot, plot_out)

    def copy_table(self, input_file, out_path):
        with open_file(input_file, 'r') as f:
            columns, table = self.read_table(f, self.summary_table_key)
        table = pd.DataFrame(table, columns=columns)
        table.to_csv(out_path, index=False)

    def compile_message(self, out_folder, plate_name, runtime):
        msg_path = os.path.join(self.out_folder, 'message.txt')
        with open(msg_path, 'w') as f:
            f.write("Results for plate %s:\n" % plate_name)
            if runtime is not None:
                # TODO format to hours
                f.write("Ran in %f s \n" % runtime)

    def post_slack(self, out_folder):
        # get slack client and connect to the workspace
        client = SlackClient(self.slack_token)
        if client.rtm_connect(with_team_state=False):
            logger.info(f"{self.name}: successfully connected to workspace")
        else:
            logger.info(f"{self.name}: connection to workspace failed")

    def run(self, input_files, output_files, runtime=None):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        plate_name = os.path.splot(self.folder)[1]
        # make the output summary folder
        out_folder = self.out_folder
        os.makedirs(out_folder, exist_ok=True)

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
