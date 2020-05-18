import os
import shutil
from glob import glob
from numbers import Number

from slack import WebClient
from slack.errors import SlackApiError

from ..analysis.cell_level_analysis import _get_bg_correction_dict
from ..base import BatchJobOnContainer
from ..config import get_default_extension
from ..util import get_logger, open_file, in_file_to_image_name
from .export_tables import export_default_table

DEFAULT_PLOT_PATTERNS = ('*ratio_of_q0.5_of_means*per-well.png',
                         '*robust_z_score_means*per-well.png')

DEFAULT_SLACK_CHANNEL = '#latest-results'
# DEFAULT_SLACK_CHANNEL = '#random'

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
            export_default_table(f, self.summary_table_key, out_path)
            param_cols, param_table = self.read_table(f, self.parameter_table_key)

        # get the parameter dict
        params = {name: val for name, val in zip(param_cols, param_table.squeeze())}
        return params

    def compile_message(self, out_folder, input_file, plate_name, runtime, background_dict):
        msg_path = os.path.join(self.out_folder, 'message.txt')

        ext = '*' + get_default_extension()
        input_pattern = os.path.join(self.input_folder, ext)
        in_files = glob(input_pattern)
        im_name = in_file_to_image_name(in_files[0])

        with open(msg_path, 'w') as f:
            f.write(f"Results for plate `{plate_name}`:\n")
            f.write(f"Files are stored in `{self.folder}`\n")
            if runtime is not None:
                f.write(f"Computation ran in {runtime} s\n")

            f.write("The following values were used for the background subtraction for the individual channels:\n")
            for channel_name, bg_key in background_dict.items():
                if isinstance(bg_key, Number):
                    f.write(f"{channel_name}: the fixed background was fixed to the value {bg_key}\n")
                else:
                    bg_src, bg_type = bg_key.split('/')
                    if bg_src == 'plate':
                        bg_col_name = f"{channel_name}_median"
                        bg_val = _get_bg_correction_dict(input_file, bg_key, bg_col_name, in_files)[im_name]
                        if bg_type == "backgrounds_from_min_well":
                            well_col_name = f"{channel_name}_min_well"
                            min_well = _get_bg_correction_dict(input_file, bg_key, well_col_name, in_files)[im_name]
                            msg = f"the background was computed via median over the minimum well {min_well}: {bg_val}"
                        else:
                            msg = f"the background was computed via median over the whole plate: {bg_val}"
                    else:
                        msg = f"the background was computed per {bg_src}"
                    f.write(f"{channel_name}: {msg}\n")

    def post_slack(self, out_folder):
        msg_sent_path = os.path.join(self.out_folder, 'message_sent.txt')

        # load the messages
        msg_path = os.path.join(self.out_folder, 'message.txt')
        with open(msg_path) as f:
            message = f.read()

        image_paths = glob(os.path.join(out_folder, '*.png'))
        image_paths.sort()
        table_path = glob(os.path.join(out_folder, '*.xlsx'))
        assert len(table_path) == 1
        table_path = table_path[0]

        # get slack client and connect to the workspace
        client = WebClient(token=self.slack_token)

        # NOTE: slack does not support multiple file upload in a single command for now,
        # neither via BlockKit (only supports remote files, not uploading local ones)
        # nor via files_upload (only supports a single file)
        # this is a bit of a pity, because it make

        # post summary to slack
        try:
            logger.info(f"{self.name}: successfully connected to workspace")
            message_blocks = [
                {
                    "type": "section",
                    "text": {"text": message, "type": "mrkdwn"}
                }
            ]
            response = client.chat_postMessage(channel=self.slack_channel,
                                               blocks=message_blocks)
            logger.debug(f"{self.name}: posted {response} to {self.slack_channel}")

            for im_path in image_paths:
                response = client.files_upload(channels=self.slack_channel, file=im_path)
                logger.debug(f"{self.name}: posted {response} to {self.slack_channel}")

            # we just write this file if the slack message was sent
            with open(msg_sent_path, 'w'):
                pass

        except SlackApiError as e:
            logger.info(f"{self.name}: connection failed with {e.response['error']}")

    def run(self, input_files, output_files, runtime=None):
        if len(input_files) != 1 or len(output_files) != 1:
            raise ValueError(f"{self.name}: expect only a single table file, not {len(input_files)}")

        input_table = input_files[0]
        plate_name = os.path.split(self.folder)[1]
        # make the output summary folder
        out_folder = self.out_folder
        os.makedirs(out_folder, exist_ok=True)
        logger.info(f"{self.name}: created summary folder {out_folder}")

        # copy the relevant plots to the summary folder
        plot_folder = os.path.join(self.folder, 'plots')
        self.copy_plots(plot_folder, out_folder)

        # copy the well table as csv to the summary folder and load the analysis parameter
        table_out_path = os.path.join(out_folder, '%s_summary.xlsx' % plate_name)
        params = self.copy_table(input_table, table_out_path)

        # compile the slack message
        background_dict = {k.lstrip('background_'): v for k, v in params.items() if k.startswith('background')}
        self.compile_message(out_folder, input_table, plate_name, runtime, background_dict)

        # post to slack with the slack bot
        if self.slack_token is None:
            logger.info(f"{self.name}: no slack token given")
        else:
            self.post_slack(out_folder)
